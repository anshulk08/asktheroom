import SwiftUI

@main
struct AskTheRoomApp: App {
    var body: some Scene {
        WindowGroup {
            PlaceholderView()
        }
    }
}

/// Stand-in until the Room screen (M1) lands: lists the mock snapshot.
struct PlaceholderView: View {
    private let snapshot = MockData.sampleSnapshot

    var body: some View {
        NavigationStack {
            List(snapshot.entities) { entity in
                HStack {
                    Text(entity.displayName)
                    Spacer()
                    Text(entity.status.rawValue).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Ask the Room")
        }
    }
}

#Preview {
    PlaceholderView()
}
