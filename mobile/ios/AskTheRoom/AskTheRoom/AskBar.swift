import SwiftUI

/// Suggestion chips (spec section 5, with the neutral pill wording from PROTOCOL_PROPOSALS.md).
struct SuggestionChips: View {
    static let suggestions = ["Where are my keys?", "What changed?", "When did I last pick up my pills?"]
    var onPick: (String) -> Void

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(Self.suggestions, id: \.self) { s in
                    Button(s) { onPick(s) }
                        .font(.subheadline.weight(.medium))
                        .buttonStyle(.bordered)
                        .buttonBorderShape(.capsule)
                        .tint(.primary)
                }
            }
            .padding(.horizontal, 16)
        }
    }
}

/// Suggestion chips and the ask bar, wired to dictation. Used on Home and on the Table screen.
struct AskPanel: View {
    var showSuggestions = true
    var onAsk: (String) -> Void
    @State private var draft = ""
    @State private var dictation = Dictation()

    var body: some View {
        VStack(spacing: 10) {
            if showSuggestions { SuggestionChips(onPick: onAsk) }
            AskBar(text: $draft,
                   isListening: dictation.isListening,
                   micAvailable: dictation.isAvailable,
                   onSend: onAsk,
                   onMicDown: {
                       Task {
                           await dictation.start(onPartial: { draft = $0 }, onFinish: { heard in
                               draft = ""
                               onAsk(heard)
                           })
                       }
                   },
                   onMicUp: { dictation.stop() })
        }
        .padding(.vertical, 10)
    }
}

/// Text field plus a large hold-to-talk mic. Dictation fills the field live and sends on release.
struct AskBar: View {
    @Binding var text: String
    var isListening = false
    var micAvailable = true
    var onSend: (String) -> Void
    var onMicDown: () -> Void = {}
    var onMicUp: () -> Void = {}

    @FocusState private var focused: Bool

    var body: some View {
        HStack(spacing: 10) {
            TextField(isListening ? "Listening…" : "Ask the room…", text: $text)
                .font(.body)
                .submitLabel(.send)
                .focused($focused)
                .onSubmit(send)
                .padding(.horizontal, 16)
                .frame(minHeight: 50)
                .background(Capsule().fill(Color(.secondarySystemBackground)))

            if !text.isEmpty && !isListening {
                Button(action: send) {
                    Image(systemName: "arrow.up")
                        .font(.title2.weight(.bold))
                        .frame(width: 56, height: 56)
                        .background(Circle().fill(Theme.laser))
                        .foregroundStyle(.white)
                }
                .accessibilityLabel("Ask")
            } else if micAvailable {
                MicButton(isListening: isListening, onDown: {
                    focused = false
                    onMicDown()
                }, onUp: onMicUp)
            }
        }
        .padding(.horizontal, 16)
    }

    private func send() {
        let q = text
        text = ""
        focused = false
        onSend(q)
    }
}

private struct MicButton: View {
    let isListening: Bool
    let onDown: () -> Void
    let onUp: () -> Void
    @GestureState private var pressed = false

    var body: some View {
        Image(systemName: isListening ? "waveform" : "mic.fill")
            .font(.title2.weight(.bold))
            .frame(width: 56, height: 56)
            .background(Circle().fill(isListening ? Theme.laser : Color.primary.opacity(0.85)))
            .foregroundStyle(isListening ? Color.white : Color(.systemBackground))
            .scaleEffect(pressed ? 1.12 : 1)
            .animation(.easeOut(duration: 0.15), value: pressed)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .updating($pressed) { _, state, _ in state = true }
            )
            .onChange(of: pressed) { _, down in down ? onDown() : onUp() }
            .accessibilityLabel(isListening ? "Stop and ask" : "Hold to ask by voice")
            .accessibilityAddTraits(.isButton)
            .accessibilityAction { isListening ? onUp() : onDown() }
    }
}

#Preview {
    VStack {
        Spacer()
        SuggestionChips { _ in }
        AskBar(text: .constant(""), onSend: { _ in })
    }
}
