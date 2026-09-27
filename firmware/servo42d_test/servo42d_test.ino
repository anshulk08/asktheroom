// Unloaded STEP/DIR test for two MKS SERVO42D drivers (pan + tilt).
//
// Wiring (COM on each driver = Uno 5V, common-anode optocoupler inputs):
//   Pan : EN=D5  STP=D6  DIR=D7
//   Tilt: EN=D2  STP=D3  DIR=D4
// (The driver on D2-D4 turned out to be the tilt motor, so the axes are
// swapped here rather than rewiring.)
// Driver settings assumed: En=L (LOW enables), MStep=16, Dir=CW.
//
// Serial Monitor @ 115200:
//   1 = pan test move out, then back
//   2 = tilt test move out, then back
//   x = stop and release both motors
//   v<us> = cruise pulse period in microseconds (smaller = faster), e.g. v1000
//   n<pulses> = pulses per test move (3200 = one motor turn), e.g. n1600
//   ? = print current settings
// Nothing moves on startup.

const uint8_t EN_PINS[2]   = {5, 2};
const uint8_t STEP_PINS[2] = {6, 3};
const uint8_t DIR_PINS[2]  = {7, 4};
const char *AXIS_NAMES[2]  = {"PAN", "TILT"};

const uint16_t RAMP_PULSES      = 200;   // pulses spent accelerating / decelerating
const uint16_t SLOW_PERIOD_US   = 5000;  // start/end pulse period
const uint16_t MIN_PERIOD_US    = 150;
const uint16_t MAX_PULSES       = 6400;
const uint8_t  PULSE_LOW_US     = 20;    // STEP active-low width
const uint16_t ENABLE_SETTLE_MS = 10;
const uint16_t REVERSE_PAUSE_MS = 500;

uint16_t testPulses   = 800;   // 90 deg motor shaft at 1.8 deg / 16 microsteps
uint16_t fastPeriodUs = 1000;  // cruise pulse period

bool stopRequested = false;

void releaseAll() {
  for (uint8_t i = 0; i < 2; i++) {
    digitalWrite(EN_PINS[i], HIGH);
  }
}

// Returns true if an 'x' arrived. Any other characters are discarded.
bool checkStop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'x' || c == 'X') {
      stopRequested = true;
    }
  }
  return stopRequested;
}

uint16_t periodFor(uint16_t i, uint16_t total) {
  uint16_t fromEdge = min(i, (uint16_t)(total - 1 - i));
  if (fromEdge >= RAMP_PULSES || fastPeriodUs >= SLOW_PERIOD_US) {
    return fastPeriodUs;
  }
  return SLOW_PERIOD_US - (uint32_t)(SLOW_PERIOD_US - fastPeriodUs) * fromEdge / RAMP_PULSES;
}

// Returns false if stopped early.
bool movePulses(uint8_t axis, bool forward, uint16_t pulses) {
  digitalWrite(DIR_PINS[axis], forward ? HIGH : LOW);
  delayMicroseconds(50);  // DIR setup time before first step

  for (uint16_t i = 0; i < pulses; i++) {
    if (checkStop()) {
      return false;
    }
    uint16_t period = periodFor(i, pulses);
    digitalWrite(STEP_PINS[axis], LOW);
    delayMicroseconds(PULSE_LOW_US);
    digitalWrite(STEP_PINS[axis], HIGH);
    delayMicroseconds(period - PULSE_LOW_US);
  }
  return true;
}

void runTest(uint8_t axis) {
  stopRequested = false;
  Serial.print(AXIS_NAMES[axis]);
  Serial.print(F(": "));
  Serial.print(testPulses);
  Serial.print(F(" pulses out, then "));
  Serial.print(testPulses);
  Serial.print(F(" back at "));
  Serial.print(fastPeriodUs);
  Serial.println(F(" us/pulse."));

  digitalWrite(EN_PINS[axis], LOW);
  delay(ENABLE_SETTLE_MS);

  bool ok = movePulses(axis, true, testPulses);
  if (ok) {
    unsigned long pauseStart = millis();
    while (millis() - pauseStart < REVERSE_PAUSE_MS) {
      if (checkStop()) {
        ok = false;
        break;
      }
    }
  }
  if (ok) {
    ok = movePulses(axis, false, testPulses);
  }

  if (ok) {
    Serial.println(F("Done. Motor left enabled; send x to release."));
  } else {
    releaseAll();
    Serial.println(F("STOPPED: both motors released."));
  }
}

void setup() {
  for (uint8_t i = 0; i < 2; i++) {
    digitalWrite(EN_PINS[i], HIGH);  // set level before switching to output
    digitalWrite(STEP_PINS[i], HIGH);
    pinMode(EN_PINS[i], OUTPUT);
    pinMode(STEP_PINS[i], OUTPUT);
    pinMode(DIR_PINS[i], OUTPUT);
  }

  Serial.begin(115200);
  Serial.setTimeout(100);
  Serial.println(F("Unloaded test: 1=pan, 2=tilt, x=stop/release. No automatic movement."));
}

void printSettings() {
  Serial.print(F("pulses="));
  Serial.print(testPulses);
  Serial.print(F(" period_us="));
  Serial.println(fastPeriodUs);
}

void loop() {
  if (!Serial.available()) {
    return;
  }
  char c = Serial.read();
  if (c == '1') {
    runTest(0);
  } else if (c == '2') {
    runTest(1);
  } else if (c == 'x' || c == 'X') {
    releaseAll();
    Serial.println(F("Both motors released."));
  } else if (c == 'v') {
    fastPeriodUs = constrain(Serial.parseInt(), MIN_PERIOD_US, SLOW_PERIOD_US);
    printSettings();
  } else if (c == 'n') {
    testPulses = constrain(Serial.parseInt(), 1, MAX_PULSES);
    printSettings();
  } else if (c == '?') {
    printSettings();
  }
}
