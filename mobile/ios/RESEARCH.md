# Design research: voice, object icons, map

Notes behind the iPhone app's choices for three questions. Each part lists the options, what they cost, and what we picked for this app: short answers, a person living with dementia, a helper setting it up, and a feature freeze on Sat Sep 26 at 6 PM EDT.

## 1. The phone's voice

The rig answers in one or two short sentences (AGENTS.md). The phone reads the same answer when "Read answers aloud" is on.

| Option | Good | Costs | Verdict |
| --- | --- | --- | --- |
| **Grok TTS, REST** (`POST https://api.x.ai/v1/tts`) | One request per answer and the reply is audio bytes that `AVAudioPlayer` plays as is. Voice, `speed` (0.7–1.5), codec, sample rate and bit rate are all settable. The voices are listed by `GET /v1/tts/voices`. About $4.20 per million characters at launch. | Needs the internet and a key. The whole clip arrives before playback starts. | **Default.** Answers are short, so waiting for the whole clip costs little and the code stays simple. |
| Grok TTS, WebSocket (`wss://api.x.ai/v1/tts`) | Starts speaking sooner. | Needs `URLSessionWebSocketTask` plus `AVAudioEngine` with PCM chunks. Route changes reset the engine and clear anything scheduled, so playback has to be tracked and rebuilt. | Not now. It only pays off for long text. |
| Grok Voice Agent (`wss://api.x.ai/v1/realtime`) | Full speech to speech. | It would answer on its own, bypassing the rig's answerer, its world model and the pill-wording filter. | **Rejected.** The phone must say what the rig decided. |
| ElevenLabs (the rig's voice, `voice/tts.py`) | Sounds exactly like the room. | Separate key and account. | Kept as a choice ("Same as the rig"). |
| iPhone voice (`AVSpeechSynthesizer`) | Offline, free, private, instant. | Sounds less natural. | **Always the fallback**: no key, offline, or no reply within 3 s. |

**Adapting to the speaker.** The phone watches `AVAudioSession` route changes and matches the voice to where the sound is going, following Apple's route-change guidance:

| Output | Speed | Audio asked for | Why |
| --- | --- | --- | --- |
| iPhone speaker | the chosen speed × 0.92 | MP3 24 kHz, 64 kbps | A small speaker at arm's length: slower is clearer, and higher fidelity is wasted. |
| Headphones, AirPods, Bluetooth music (A2DP/LE) | as chosen | MP3 44.1 kHz, 128 kbps | Close to the ear and full bandwidth. |
| Bluetooth headset call profile (HFP) | as chosen | MP3 16 kHz, 32 kbps | The link itself is 16 kHz, so more is wasted. |
| AirPlay, HDMI, car, USB speakers | the chosen speed × 0.9 | MP3 48 kHz, 192 kbps | Room speakers: slower carries better across a room. |

When headphones or Bluetooth disconnect in the middle of an answer, speech stops. It doesn't carry on out loud through the speaker; Apple's rule is to pause when a device goes away and carry on when one is added. The session is `.playback` with `.spokenAudio` and `.duckOthers`. That category routes to A2DP without asking and never uses the earpiece.

**Keys.** The xAI and ElevenLabs keys are entered in Helper settings and stored only in the phone's Keychain (`WhenUnlockedThisDeviceOnly`). They are never in the repo or the app bundle. With Grok on, the answer text goes to xAI to be spoken.

## 2. Icons for the objects

We want the map to read like Find My, where every item is a round pin with its own picture.

| Option | Good | Costs | Verdict |
| --- | --- | --- | --- |
| **Emoji** (🔑 👛 👓 📱 💊 📓 📦) | Colourful and concrete, like Find My's AirTag pins. The same everywhere, at no cost. Dementia guidance favours realistic pictures over abstract marks. | Can't be restyled, and some objects have none (a TV remote). | **Default where a good one exists.** |
| **SF Symbols** | Matches the system, works in dark mode, scales with Dynamic Type, and has 7,000+ symbols (`appletvremote.gen4.fill`, `powerplug.fill`). | Monochrome and more abstract. | **Fallback** for objects with no emoji, and on pin badges (hand, question mark, link). |
| **Genmoji** (`NSAdaptiveImageGlyph`, iOS 18+) | A custom picture for anything ("my charger", a pill organiser). Made by the helper from the keyboard's Genmoji tab. Stored as HEIC data with alt text. | Making one needs an Apple Intelligence iPhone (15 Pro or later). SwiftUI's `TextField` can't accept them, so it needs a `UITextView` with `supportsAdaptiveImageGlyph`. An app can't generate one in code. | **Optional override** in "Change picture". Shown on any iOS 18+ device. |
| Image Playground (`imagePlaygroundSheet`) | Generates from a text concept in code. | Needs an Apple Intelligence device and doesn't work in the simulator. It produces a full illustration, which is busy at 40 pt. It's the same engine as Genmoji, and Genmoji's emoji style suits pins better. | Not used. |
| Bundled custom illustrations | Full control. | Design time we don't have before the freeze. | Not now. |

So each thing shows a custom Genmoji or emoji if the helper set one, then the built-in emoji, then an SF Symbol. The icon on Home, the cards and the map is the same everywhere, so the picture the person learns is the one they see on the map.

## 3. The map

Find My shows a round pin with a picture, fades a pin whose location is stale, and draws a pale ring when the location is uncertain. We copy that, but add the name under each pin, because dementia guidance says icons always come with words.

- **Pin:** a white (or dark) disc with a soft shadow and the picture, at least 44 pt to tap.
- **Status as small badges on the pin's corner**, the way Find My and Messages badge avatars: a hand when held, a question mark when the room can't see it, a link when it might be an older thing. A dashed ring means hidden inside or under something. A faded pin means not sure. An arrow means it left the table.
- **Containers and covers stay drawn to scale**, with their picture and name inside, so "inside the box" is visible as a pin sitting inside the box.
- **Selection:** the pin grows a little, gets an accent ring, and gives a light selection haptic (`sensoryFeedback`). Springs are skipped with Reduce Motion.
- **We don't use MapKit.** This is a table in centimetres, not the Earth. SwiftUI's `Map`/`Annotation` would bring tiles and geographic coordinates we'd have to fake, and custom annotations don't cluster. The map stays a plain SwiftUI view with our own placement.

## Sources

- xAI: [Text to Speech](https://docs.x.ai/developers/model-capabilities/audio/text-to-speech), [Voice REST reference](https://docs.x.ai/developers/rest-api-reference/inference/voice), [Grok STT and TTS APIs](https://x.ai/news/grok-stt-and-tts-apis), [Voice Agent API](https://x.ai/news/grok-voice-agent-api), [pricing note (openclaw issue)](https://github.com/openclaw/openclaw/issues/48454)
- Apple audio: [Responding to route changes](https://developer.apple.com/library/content/documentation/Audio/Conceptual/AudioSessionProgrammingGuide/HandlingAudioHardwareRouteChanges/HandlingAudioHardwareRouteChanges.html), [QA1803 categories and AirPlay](https://developer.apple.com/library/ios/qa/qa1803/_index.html), [Understanding AVAudioSession routes](https://medium.com/@mehsamadi/understanding-avaudiosession-routes-on-ios-7718d934d0c0)
- Genmoji and Image Playground: [WWDC24 Bring expression to your app with Genmoji](https://developer.apple.com/videos/play/wwdc2024/10220/), [Genmoji and NSAdaptiveImageGlyph](https://blakecrosley.com/blog/genmoji-nsadaptiveimageglyph), [Enabling Genmoji in your app](https://www.createwithswift.com/enabling-genmoji-in-your-app/), [imagePlaygroundSheet](https://developer.apple.com/documentation/swiftui/view/imageplaygroundsheet(ispresented:concept:sourceimageurl:oncompletion:oncancellation:)), [Hacking with Swift: Image Playground](https://www.hackingwithswift.com/quick-start/swiftui/how-to-generate-images-using-image-playground)
- Find My and maps: [AirTag emoji](https://9to5mac.com/2021/05/01/how-to-pick-a-custom-emoji-and-name-for-your-airtag/), [Find My icons explained](https://techwiser.com/what-find-my-app-icons-mean-complete-guide/), [SwiftUI Map annotations](https://www.hackingwithswift.com/quick-start/swiftui/how-to-show-annotations-in-a-map-view), [MapKit clustering](https://developer.apple.com/documentation/MapKit/decluttering-a-map-with-mapkit-annotation-clustering), [SF Symbols](https://developer.apple.com/sf-symbols/)
- Cognitive accessibility: [W3C Avoid too much content](https://www.w3.org/WAI/WCAG2/supplemental/patterns/o5p03-manageable-quantity/), [Dementia digital design guidelines](https://rikwilliams.net/resources/dementia-digital-design-guidelines/), [AbilityNet: designing for dementia](https://abilitynet.org.uk/factsheets/designing-dementia)
