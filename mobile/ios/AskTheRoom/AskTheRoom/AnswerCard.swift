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
                    Text("Asking the room…").foregroundStyle(.secondary)
                }
                .font(.title3)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 18))
    }
}

/// An answer to a question someone spoke to the rig (PROTOCOL_PROPOSALS.md P2; off by default).
struct HeardInRoomCard: View {
    let answer: Answer

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Heard in the room", systemImage: "waveform")
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

/// Earlier questions and answers, newest first.
struct HistoryList: View {
    let exchanges: ArraySlice<Exchange>

    var body: some View {
        if !exchanges.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                Text("Earlier")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.secondary)
                ForEach(exchanges) { e in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(e.question).font(.footnote).foregroundStyle(.secondary)
                        Text(e.answer?.text ?? (e.timedOut ? "No answer" : "…"))
                            .font(.body)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .accessibilityElement(children: .combine)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 4)
        }
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
