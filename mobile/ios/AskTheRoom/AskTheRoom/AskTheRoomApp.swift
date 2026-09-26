import SwiftUI

@main
struct AskTheRoomApp: App {
    /// `-mock YES` starts in mock mode; see MockRoom for the other launch arguments.
    @State private var store = RoomStore(mock: UserDefaults.standard.bool(forKey: "mock"),
                                         liveTransport: { RoomLink(store: $0) })

    var body: some Scene {
        WindowGroup {
            RootView(store: store)
        }
    }
}
