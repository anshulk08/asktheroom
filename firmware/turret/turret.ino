// Laser pointing turret: Arduino Uno R3 + two MKS SERVO42D (STEP/DIR) on the
// Instructables pan/tilt mount. A host (Jetson or PC) drives it over USB serial.
//
// Serial: 115200 baud, one command per line ('\n' terminated), case-insensitive.
//   A <pan> <tilt>    aim at absolute angles in degrees (both axes move together)
//   R <dpan> <dtilt>  move relative to the current target
//   L <0|1>           laser off / on
//   B <0-255>         laser brightness while on (PWM pins only; elsewhere any level > 0 is full on)
//   S                 smooth stop (decelerate, keep holding)
//   X                 emergency stop: pulses stop, laser off, motors released
//   E <0|1>           release / enable motors
//   Z                 call the current position 0,0 (only while stopped)
//   V <deg/s>         max axis speed
//   C <deg/s^2>       axis acceleration
//   P or ?            report "POS <pan> <tilt> <moving> <laser>"
//   H                 list commands
// Replies: "OK ..." or "ERR ...". "DONE" is sent when a move finishes.
// Boot prints "READY". Opening the port resets the Uno, so position 0,0 is
// wherever the mount is at that moment.
//
// Safety: the laser turns itself off if no command arrives for
// LASER_TIMEOUT_MS. A host that wants it to stay on sends P as a heartbeat.

#include <util/atomic.h>

// ---- Wiring -----------------------------------------------------------------
// Driver inputs are common-anode optocouplers with COM on Uno 5V, so LOW = active.
// All six driver pins are on PORTD so the step interrupt can use direct port writes.
const uint8_t EN_PIN[2]   = {5, 2};  // pan, tilt
const uint8_t STEP_PIN[2] = {6, 3};
const uint8_t DIR_PIN[2]  = {7, 4};
const uint8_t STEP_MASK[2] = {_BV(PD6), _BV(PD3)};
const uint8_t LASER_PIN = 8;         // laser module signal (S); not a PWM pin, so on/off only
const char *AXIS_NAME[2] = {"pan", "tilt"};

// ---- Mechanics --------------------------------------------------------------
// 1.8 deg motors at MStep=16; 20T motor pulleys. The build guide lists 64T and
// 80T driven pulleys; on this mount pan is 80T (4:1), verified with +/-90 deg swings.
// Tilt's 64T (3.2:1) is inferred, not yet measured.
const float MOTOR_STEPS_PER_REV = 200.0 * 16;
const float GEAR_RATIO[2] = {4.0, 3.2};
const bool  INVERT_DIR[2] = {true, true};    // + pans right, + tilts up
const float MIN_DEG[2] = {-90, -90};   // the mount's travel: 90 deg each way on both axes
const float MAX_DEG[2] = { 90,  90};

// ---- Motion -----------------------------------------------------------------
const uint16_t TICK_HZ = 20000;             // step interrupt rate
const float MAX_STEP_RATE = TICK_HZ / 2.0;  // a pulse is one tick low, one tick high
const float MIN_STEP_RATE = 100;            // creep speed so every move finishes
const float ARRIVE_STEPS = 4;
const uint16_t PLAN_US = 1000;              // velocity planner period
const uint16_t ENABLE_SETTLE_MS = 10;
const uint32_t LASER_TIMEOUT_MS = 2000;

float maxSpeedDeg = 120;   // deg/s at the axis
float accelDeg = 600;      // deg/s^2 at the axis

float stepsPerDeg[2];
float maxRate[2];          // steps/s
float accelRate[2];        // steps/s^2
float vel[2];              // signed steps/s, owned by the planner

// Shared with the step interrupt.
volatile int32_t pos[2];
volatile int32_t tgt[2];
volatile uint16_t rate[2];     // steps per tick, Q16
volatile uint16_t phase[2];
volatile int8_t dirSign[2] = {1, 1};
volatile bool guardTarget[2];  // don't step past tgt (we're approaching it slowly enough)

bool motorsEnabled = false;
bool moveActive = false;
bool laserOn = false;
uint8_t laserBrightness = 255;
uint32_t lastCommandMs = 0;

char line[48];
uint8_t lineLen = 0;
bool lineOverflow = false;

ISR(TIMER2_COMPA_vect) {
  PORTD |= STEP_MASK[0] | STEP_MASK[1];  // end any pulse started last tick
  for (uint8_t i = 0; i < 2; i++) {
    uint16_t r = rate[i];
    if (!r) {
      continue;
    }
    uint16_t prev = phase[i];
    phase[i] = prev + r;
    if (phase[i] >= prev) {
      continue;  // no overflow, no step due
    }
    if (guardTarget[i] && pos[i] == tgt[i]) {
      continue;
    }
    PORTD &= ~STEP_MASK[i];
    pos[i] += dirSign[i];
  }
}

int32_t readPos(uint8_t i) {
  int32_t p;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { p = pos[i]; }
  return p;
}

int32_t readTgt(uint8_t i) {
  int32_t t;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { t = tgt[i]; }
  return t;
}

void writeTgt(uint8_t i, int32_t t) {
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { tgt[i] = t; }
}

void applyMotionLimits() {
  for (uint8_t i = 0; i < 2; i++) {
    maxRate[i] = constrain(maxSpeedDeg * stepsPerDeg[i], MIN_STEP_RATE, MAX_STEP_RATE * 0.9);
    accelRate[i] = max(accelDeg * stepsPerDeg[i], 100.0f);
  }
}

void setDirection(uint8_t i, int8_t s) {
  if (s == dirSign[i]) {
    return;
  }
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { rate[i] = 0; }
  digitalWrite(DIR_PIN[i], (s > 0) != INVERT_DIR[i] ? HIGH : LOW);
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    dirSign[i] = s;
    phase[i] = 0;  // first step in the new direction comes well after the DIR change
  }
}

// Trapezoidal velocity planner, run every PLAN_US. Retargeting mid-move is fine:
// it brakes, reverses if needed, and re-plans toward the new target.
void plan(uint8_t i, float dt) {
  int32_t d = readTgt(i) - readPos(i);
  float v = vel[i];
  float a = accelRate[i];
  float dv = a * dt;
  float speed = fabs(v);
  int8_t s = (d > 0) - (d < 0);
  // Steps we can afford to stop "abruptly" within; the driver's closed loop absorbs it.
  float margin = ARRIVE_STEPS + speed * dt * 2;

  if (d == 0 && v * v / (2 * a) <= margin) {
    v = 0;
  } else if (v != 0 && v * s <= 0) {
    // Moving away from the target (or through it): brake to zero before reversing.
    speed = max(speed - dv, 0.0f);
    v = v > 0 ? speed : -speed;
  } else if (labs(d) <= v * v / (2 * a)) {
    v = s * max(speed - dv, MIN_STEP_RATE);
  } else {
    v = s * min(max(speed, MIN_STEP_RATE) + dv, maxRate[i]);
  }
  vel[i] = v;

  int8_t vs = (v > 0) - (v < 0);
  if (vs != 0) {
    setDirection(i, vs);
  }
  // Only stop exactly on target when we could have stopped there anyway;
  // otherwise let it overshoot slightly and come back, instead of slamming to a halt.
  margin = ARRIVE_STEPS + fabs(v) * dt * 2;
  bool guard = vs == s && labs(d) + margin >= v * v / (2 * a);
  uint16_t r = min(fabs(v) * (65536.0 / TICK_HZ), 32768.0);
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    guardTarget[i] = guard;
    rate[i] = r;
  }
}

bool isMoving() {
  for (uint8_t i = 0; i < 2; i++) {
    if (vel[i] != 0 || readPos(i) != readTgt(i)) {
      return true;
    }
  }
  return false;
}

void enableMotors(bool on) {
  if (on == motorsEnabled) {
    return;
  }
  for (uint8_t i = 0; i < 2; i++) {
    digitalWrite(EN_PIN[i], on ? LOW : HIGH);
  }
  motorsEnabled = on;
  if (on) {
    delay(ENABLE_SETTLE_MS);
  }
}

void haltPulses() {
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    for (uint8_t i = 0; i < 2; i++) {
      rate[i] = 0;
      tgt[i] = pos[i];
    }
  }
  vel[0] = vel[1] = 0;
  moveActive = false;
}

void setLaser(bool on) {
  laserOn = on;
  if (on && digitalPinHasPWM(LASER_PIN) && laserBrightness < 255) {
    analogWrite(LASER_PIN, laserBrightness);
  } else if (on) {
    digitalWrite(LASER_PIN, laserBrightness ? HIGH : LOW);
  } else {
    digitalWrite(LASER_PIN, LOW);
  }
}

float stepsToDeg(uint8_t i, int32_t steps) {
  return steps / stepsPerDeg[i];
}

// Clamps to the soft limits and returns the degrees actually targeted.
float aimAxis(uint8_t i, float deg) {
  deg = constrain(deg, MIN_DEG[i], MAX_DEG[i]);
  writeTgt(i, lround(deg * stepsPerDeg[i]));
  return deg;
}

void startMove(float panDeg, float tiltDeg) {
  enableMotors(true);
  panDeg = aimAxis(0, panDeg);
  tiltDeg = aimAxis(1, tiltDeg);
  moveActive = true;
  Serial.print(F("OK A "));
  Serial.print(panDeg, 2);
  Serial.print(' ');
  Serial.println(tiltDeg, 2);
}

void report() {
  Serial.print(F("POS "));
  Serial.print(stepsToDeg(0, readPos(0)), 2);
  Serial.print(' ');
  Serial.print(stepsToDeg(1, readPos(1)), 2);
  Serial.print(' ');
  Serial.print(isMoving() ? 1 : 0);
  Serial.print(' ');
  Serial.println(laserOn ? 1 : 0);
}

void printHelp() {
  Serial.println(F("A pan tilt | R dpan dtilt | L 0/1 | B 0-255 | S | X | E 0/1 | Z | V deg/s | C deg/s2 | P"));
  for (uint8_t i = 0; i < 2; i++) {
    Serial.print(AXIS_NAME[i]);
    Serial.print(F(" limits "));
    Serial.print(MIN_DEG[i], 0);
    Serial.print(F(".."));
    Serial.print(MAX_DEG[i], 0);
    Serial.print(F(" deg, max "));
    Serial.print(maxRate[i] / stepsPerDeg[i], 0);
    Serial.println(F(" deg/s"));
  }
}

// Parses up to `count` floats after the command letter. Returns how many were found.
uint8_t parseFloats(const char *p, float *out, uint8_t count) {
  uint8_t n = 0;
  while (n < count) {
    char *end;
    double v = strtod(p, &end);
    if (end == p) {
      break;
    }
    out[n++] = v;
    p = end;
  }
  return n;
}

void handleCommand(char *cmd) {
  while (*cmd == ' ') {
    cmd++;
  }
  if (!*cmd) {
    return;
  }
  lastCommandMs = millis();
  char op = toupper(*cmd);
  const char *args = cmd + 1;
  float v[2];
  uint8_t n = parseFloats(args, v, 2);

  switch (op) {
    case 'A':
      if (n < 2) {
        Serial.println(F("ERR A needs pan and tilt"));
        return;
      }
      startMove(v[0], v[1]);
      break;
    case 'R':
      if (n < 2) {
        Serial.println(F("ERR R needs dpan and dtilt"));
        return;
      }
      startMove(stepsToDeg(0, readTgt(0)) + v[0], stepsToDeg(1, readTgt(1)) + v[1]);
      break;
    case 'L':
      if (n < 1) {
        Serial.println(F("ERR L needs 0 or 1"));
        return;
      }
      setLaser(v[0] != 0);
      Serial.println(laserOn ? F("OK L 1") : F("OK L 0"));
      break;
    case 'B':
      if (n < 1) {
        Serial.println(F("ERR B needs 0-255"));
        return;
      }
      laserBrightness = constrain((int)v[0], 0, 255);
      if (laserOn) {
        setLaser(true);
      }
      Serial.print(F("OK B "));
      Serial.println(laserBrightness);
      break;
    case 'S':
      for (uint8_t i = 0; i < 2; i++) {
        float stopSteps = vel[i] * vel[i] / (2 * accelRate[i]);
        writeTgt(i, readPos(i) + (vel[i] >= 0 ? 1 : -1) * (int32_t)ceil(stopSteps));
      }
      Serial.println(F("OK S"));
      break;
    case 'X':
      haltPulses();
      setLaser(false);
      enableMotors(false);
      Serial.println(F("OK X"));
      break;
    case 'E':
      if (n < 1) {
        Serial.println(F("ERR E needs 0 or 1"));
        return;
      }
      if (v[0] == 0) {
        haltPulses();
      }
      enableMotors(v[0] != 0);
      Serial.println(motorsEnabled ? F("OK E 1") : F("OK E 0"));
      break;
    case 'Z':
      if (isMoving()) {
        Serial.println(F("ERR busy"));
        return;
      }
      ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
        pos[0] = pos[1] = tgt[0] = tgt[1] = 0;
      }
      Serial.println(F("OK Z"));
      break;
    case 'V':
    case 'C':
      if (n < 1 || v[0] <= 0) {
        Serial.println(F("ERR needs a positive number"));
        return;
      }
      if (op == 'V') {
        maxSpeedDeg = v[0];
      } else {
        accelDeg = v[0];
      }
      applyMotionLimits();
      Serial.print(F("OK V "));
      Serial.print(maxRate[0] / stepsPerDeg[0], 1);
      Serial.print(' ');
      Serial.print(maxRate[1] / stepsPerDeg[1], 1);
      Serial.print(F(" C "));
      Serial.println(accelDeg, 1);
      break;
    case 'P':
    case '?':
      report();
      break;
    case 'H':
      printHelp();
      break;
    default:
      Serial.print(F("ERR unknown "));
      Serial.println(op);
  }
}

void pollSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') {
      continue;
    }
    if (c != '\n') {
      if (lineLen < sizeof(line) - 1) {
        line[lineLen++] = c;
      } else {
        lineOverflow = true;
      }
      continue;
    }
    line[lineLen] = '\0';
    if (lineOverflow) {
      Serial.println(F("ERR line too long"));
    } else {
      handleCommand(line);
    }
    lineLen = 0;
    lineOverflow = false;
  }
}

void setup() {
  for (uint8_t i = 0; i < 2; i++) {
    digitalWrite(EN_PIN[i], HIGH);  // set levels before switching to output
    digitalWrite(STEP_PIN[i], HIGH);
    pinMode(EN_PIN[i], OUTPUT);
    pinMode(STEP_PIN[i], OUTPUT);
    pinMode(DIR_PIN[i], OUTPUT);
    digitalWrite(DIR_PIN[i], INVERT_DIR[i] ? LOW : HIGH);
    stepsPerDeg[i] = MOTOR_STEPS_PER_REV * GEAR_RATIO[i] / 360.0;
  }
  digitalWrite(LASER_PIN, LOW);
  pinMode(LASER_PIN, OUTPUT);
  applyMotionLimits();

  // Timer2 CTC at 20 kHz drives step generation (disables PWM on D3/D11).
  TCCR2A = _BV(WGM21);
  TCCR2B = _BV(CS21);                  // 16 MHz / 8 = 2 MHz
  OCR2A = F_CPU / 8 / TICK_HZ - 1;
  TIMSK2 = _BV(OCIE2A);

  Serial.begin(115200);
  Serial.println(F("READY turret"));
}

void loop() {
  pollSerial();

  static uint32_t lastPlanUs = micros();
  uint32_t now = micros();
  if (now - lastPlanUs >= PLAN_US) {
    float dt = min((now - lastPlanUs) * 1e-6f, 0.01f);
    lastPlanUs = now;
    if (motorsEnabled) {
      plan(0, dt);
      plan(1, dt);
    }
    if (moveActive && !isMoving()) {
      moveActive = false;
      Serial.println(F("DONE"));
    }
  }

  if (laserOn && millis() - lastCommandMs > LASER_TIMEOUT_MS) {
    setLaser(false);
    Serial.println(F("LASER TIMEOUT"));
  }
}
