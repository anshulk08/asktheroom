import SwiftUI

/// The latest question and its answer, shown large; `text` is shown exactly as the rig sent it.
struct AnswerCard: View {
    let exchange: Exchange
    var onRetry: () -> Void = {}

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(exchange.question)
                .font(.body)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(Theme.accent.opacity(0.14), in: RoundedRectangle(cornerRadius: 14))
                .frame(maxWidth: .infinity, alignment: .trailing)
                .accessibilityLabel("You asked: \(exchange.question)")

            if let answer = exchange.answer {
                Text(answer.text)
                    .font(.title2.weight(.semibold))
                    .foregroundStyle(answer.succeeded ? .primary : .secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityAddTraits(.updatesFrequently)
            } else if exchange.timedOut {
                Text("The room didn't answer. Try again?")
                    .font(.title3.weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)
                Button("Try again", action: onRetry)
                    .buttonStyle(.borderedProminent)
            } else {
                HStack(spacing: 10) {
                    ProgressView()
                    Text(exchange.slow ? "Still working on it…" : "Asking the room…").foregroundStyle(.secondary)
                }
                .font(.title3)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 18))
    }
}

/// An answer to a question someone else asked the rig (PROTOCOL.md 6a; off by default).
struct HeardInRoomCard: View {
    let answer: Answer

    private var heading: String {
        switch answer.src {
        case "dashboard": return "Asked on the dashboard"
        case "sms": return "Asked by text message"
        default: return "Heard in the room"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(heading, systemImage: answer.src == "voice" ? "waveform" : "text.bubble")
                .font(.footnote.weight(.semibold))
                .foregroundStyle(.secondary)
            if let q = answer.q {
                Text("“\(q)”").font(.body).foregroundStyle(.secondary)
            }
            Text(answer.text)
                .font(.title3.weight(.semibold))
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 18))
    }
}

#Preview {
    VStack {
        AnswerCard(exchange: Exchange(id: 1, question: "Where are my keys?",
                                      answer: Answer(id: 1, ok: true, text: "Your keys are inside the box.")))
        AnswerCard(exchange: Exchange(id: 2, question: "Where is my wallet?", timedOut: true))
        AnswerCard(exchange: Exchange(id: 3, question: "What changed?"))
    }
    .padding()
}
