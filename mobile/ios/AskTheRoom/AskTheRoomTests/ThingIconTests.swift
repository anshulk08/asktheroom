import UIKit
import XCTest
@testable import AskTheRoom

@MainActor
final class ThingIconTests: XCTestCase {
    func testEverySampleThingGetsARealPicture() {
        XCTAssertEqual(ThingIcon.suggested(for: "keys"), .emoji("🔑"))
        XCTAssertEqual(ThingIcon.suggested(for: "pill_bottle"), .emoji("💊"))
        XCTAssertEqual(ThingIcon.suggested(for: "remote"), .symbol("appletvremote.gen4.fill"), "no emoji for a remote")
        XCTAssertEqual(ThingIcon.suggested(for: "thing:7", title: "my charger"), .emoji("🔌"))
        XCTAssertEqual(ThingIcon.suggested(for: "thing:11", title: "phone charger?"), .emoji("🔌"), "a charger, not a phone")
        XCTAssertEqual(ThingIcon.suggested(for: "sunglasses"), .emoji("🕶️"))
        XCTAssertEqual(ThingIcon.suggested(for: "thing:9", title: "something new"), .symbol("tag.fill"))
        XCTAssertEqual(ThingIcon.suggested(for: "keyboard"), .symbol("tag.fill"), "whole words only")
    }

    func testEverySymbolChoiceExists() {
        for name in ThingIcon.symbolChoices {
            XCTAssertNotNil(UIImage(systemName: name), name)
        }
    }

    func testPicksAreSavedAndCanBeUndone() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: file) }
        let defaults = try XCTUnwrap(UserDefaults(suiteName: #function))
        defaults.removePersistentDomain(forName: #function)

        let store = IconStore(file: file, defaults: defaults)
        let genmoji = ThingIcon.genmoji(Data([1, 2, 3]), description: "a pill organiser")
        store.set(genmoji, for: "thing:4")
        store.set(.emoji("📺"), for: "remote")
        XCTAssertEqual(IconStore(file: file, defaults: defaults).icon(for: "thing:4"), genmoji)
        XCTAssertEqual(IconStore(file: file, defaults: defaults).icon(for: "remote"), .emoji("📺"))

        store.set(nil, for: "remote")
        XCTAssertEqual(IconStore(file: file, defaults: defaults).icon(for: "remote"), .symbol("appletvremote.gen4.fill"))
    }
}
