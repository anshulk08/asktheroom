import SwiftUI
import UIKit

/// The picture for one thing, the same on the map, Home, Recent and the cards (RESEARCH.md
/// section 2): an emoji where a good one exists, like an AirTag in Find My; an SF Symbol where
/// none does; or a helper's own emoji or Genmoji.
enum ThingIcon: Equatable, Hashable, Codable {
    case emoji(String)
    case symbol(String)
    /// A Genmoji's HEIC image and Apple's description of it (iOS 18+ to make or paste one).
    case genmoji(Data, description: String)

    /// Words in a thing's name, first match wins, so "my charger" and "phone charger" get a plug.
    private static let byWord: [(words: [String], icon: ThingIcon)] = [
        (["key", "keys"], .emoji("🔑")),
        (["charger", "plug", "cable"], .emoji("🔌")),
        (["phone", "iphone", "mobile"], .emoji("📱")),
        (["wallet", "purse"], .emoji("👛")),
        (["sunglasses"], .emoji("🕶️")),
        (["glasses", "spectacles", "specs"], .emoji("👓")),
        (["remote"], .symbol("appletvremote.gen4.fill")),
        (["pill", "pills", "medicine", "medication", "tablets"], .emoji("💊")),
        (["box"], .emoji("📦")),
        (["notebook", "book", "diary"], .emoji("📓")),
        (["cup", "mug", "tea", "coffee"], .emoji("☕️")),
        (["watch"], .emoji("⌚️")),
        (["headphones", "earphones", "airpods"], .emoji("🎧")),
        (["hearing"], .emoji("🦻")),
        (["teeth", "dentures"], .emoji("🦷")),
        (["pen", "pencil"], .emoji("✏️")),
        (["umbrella"], .emoji("☂️")),
        (["bag", "handbag"], .emoji("👜")),
        (["card", "cards"], .emoji("💳")),
        (["camera"], .emoji("📷")),
        (["hat", "cap"], .emoji("🧢")),
        (["scissors"], .emoji("✂️")),
    ]

    /// The picture a thing gets until a helper picks another.
    static func suggested(for name: String, title: String? = nil) -> ThingIcon {
        let words = [name, title ?? ""].joined(separator: " ").lowercased()
            .split(whereSeparator: { !$0.isLetter }).map(String.init)
        for (keys, icon) in byWord where keys.contains(where: words.contains) {
            return icon
        }
        return .symbol("tag.fill")
    }

    /// Everyday things a helper can pick with one tap, without the keyboard.
    static let emojiChoices = ["🔑", "👛", "👓", "🕶️", "📱", "💊", "📦", "📓", "🔌", "☕️", "⌚️", "🎧",
                               "🦻", "🦷", "✏️", "☂️", "👜", "💳", "📷", "🧢", "🧣", "🧤", "📺", "🪥"]
    /// For things with no emoji.
    static let symbolChoices = ["appletvremote.gen4.fill", "key.fill", "iphone", "wallet.bifold.fill",
                                "eyeglasses", "pills.fill", "shippingbox.fill", "book.closed.fill",
                                "powerplug.fill", "cup.and.saucer.fill", "headphones", "applewatch",
                                "ear.fill", "pencil", "umbrella.fill", "bag.fill", "creditcard.fill",
                                "camera.fill", "gamecontroller.fill", "tag.fill"]
}

/// Helpers' picks, kept on this phone. Names are the rig's (`keys`, `thing:7`).
@Observable @MainActor
final class IconStore {
    static let shared = IconStore()

    private(set) var custom: [String: ThingIcon] = [:]
    private let file: URL?

    init(file: URL? = IconStore.defaultFile, defaults: UserDefaults = .standard) {
        self.file = file
        if let file, let data = try? Data(contentsOf: file) {
            custom = (try? JSONDecoder().decode([String: ThingIcon].self, from: data)) ?? [:]
        }
        // -mockIcons "remote=📺|thing:9=🧸" for screenshots; not saved.
        for pair in defaults.string(forKey: "mockIcons")?.split(separator: "|") ?? [] {
            let parts = pair.split(separator: "=", maxSplits: 1).map(String.init)
            if parts.count == 2 { custom[parts[0]] = .emoji(parts[1]) }
        }
    }

    nonisolated static var defaultFile: URL? {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first?
            .appendingPathComponent("thing-icons.json")
    }

    func icon(for name: String, title: String? = nil) -> ThingIcon {
        custom[name] ?? .suggested(for: name, title: title ?? Entity.displayName(for: name))
    }

    /// nil goes back to the suggested picture.
    func set(_ icon: ThingIcon?, for name: String) {
        custom[name] = icon
        guard let file, let data = try? JSONEncoder().encode(custom) else { return }
        try? FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: file, options: [.atomic, .completeFileProtection])
    }
}

/// Draws a `ThingIcon` in a square of `size` points. Decorative: the name is always next to it.
struct ThingIconView: View {
    let icon: ThingIcon
    var size: CGFloat

    var body: some View {
        Group {
            switch icon {
            case .emoji(let emoji):
                Text(emoji).font(.system(size: size * 0.8))
            case .symbol(let name):
                Image(systemName: name)
                    .font(.system(size: size * 0.6, weight: .semibold))
                    .foregroundStyle(.primary)
            case .genmoji(let data, _):
                if let image = GenmojiCache.image(for: data) {
                    Image(uiImage: image).resizable().scaledToFit()
                } else {
                    Image(systemName: "tag.fill").font(.system(size: size * 0.6, weight: .semibold))
                }
            }
        }
        .frame(width: size, height: size)
        .accessibilityHidden(true)
    }
}

/// Decoding a Genmoji's HEIC on every redraw of the map would be wasteful.
private enum GenmojiCache {
    private static let cache = NSCache<NSData, UIImage>()

    static func image(for data: Data) -> UIImage? {
        if let hit = cache.object(forKey: data as NSData) { return hit }
        guard let image = UIImage(data: data) else { return nil }
        cache.setObject(image, forKey: data as NSData)
        return image
    }
}

/// For a helper: choose the picture for one thing. One tap for common emoji and symbols, or
/// the emoji keyboard for anything else, including a Genmoji on iPhones that can make them.
struct IconPicker: View {
    let name: String
    let title: String
    @Environment(\.dismiss) private var dismiss
    @State private var typing = false
    private let store = IconStore.shared

    private let columns = [GridItem(.adaptive(minimum: 52), spacing: 8)]

    var body: some View {
        NavigationStack {
            List {
                Section {
                    VStack(spacing: 8) {
                        ThingIconView(icon: store.icon(for: name, title: title), size: 64)
                            .frame(width: 96, height: 96)
                            .background(Circle().fill(Color(.systemBackground)).shadow(color: .black.opacity(0.15), radius: 4, y: 2))
                        Text(Dashboard.capitalized(title)).font(.title3.bold())
                    }
                    .frame(maxWidth: .infinity)
                    .listRowBackground(Color.clear)
                }

                Section {
                    GenmojiField(isEditing: $typing) { store.set($0, for: name) }
                        .frame(height: 56)
                        .overlay {
                            if !typing {
                                Label("Tap for the emoji keyboard", systemImage: "face.smiling")
                                    .foregroundStyle(.secondary)
                                    .allowsHitTesting(false)
                                    .accessibilityHidden(true)
                            }
                        }
                } header: {
                    Text("Any emoji or Genmoji")
                } footer: {
                    Text("Tap the box and pick from the emoji keyboard. On an iPhone with Apple Intelligence, type what it looks like there to make a Genmoji.")
                }

                Section("Emoji") {
                    grid(ThingIcon.emojiChoices.map(ThingIcon.emoji))
                }
                Section("Symbols") {
                    grid(ThingIcon.symbolChoices.map(ThingIcon.symbol))
                }

                if store.custom[name] != nil {
                    Section {
                        Button("Use the usual picture") { store.set(nil, for: name) }
                    }
                }
            }
            .navigationTitle("Picture")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func grid(_ icons: [ThingIcon]) -> some View {
        let current = store.icon(for: name, title: title)
        return LazyVGrid(columns: columns, spacing: 8) {
            ForEach(icons, id: \.self) { icon in
                Button { store.set(icon, for: name) } label: {
                    ThingIconView(icon: icon, size: 30)
                        .frame(width: 52, height: 52)
                        .background(RoundedRectangle(cornerRadius: 12).fill(Color(.tertiarySystemFill)))
                        .overlay {
                            if icon == current {
                                RoundedRectangle(cornerRadius: 12).strokeBorder(Theme.accent, lineWidth: 3)
                            }
                        }
                }
                .buttonStyle(.plain)
                .accessibilityLabel(label(for: icon))
                .accessibilityAddTraits(icon == current ? .isSelected : [])
            }
        }
        .padding(.vertical, 6)
        .sensoryFeedback(.selection, trigger: current)
    }

    private func label(for icon: ThingIcon) -> String {
        switch icon {
        case .emoji(let e): return e.unicodeScalars.first?.properties.name?.capitalized ?? e
        case .symbol(let s): return s.split(separator: ".").first.map(String.init) ?? s
        case .genmoji(_, let description): return description
        }
    }
}

/// SwiftUI's TextField can't take a Genmoji, so this wraps a UITextView that can (iOS 18+).
/// Each emoji or Genmoji typed becomes the pick; the field then clears for the next try.
private struct GenmojiField: UIViewRepresentable {
    @Binding var isEditing: Bool
    var onPick: (ThingIcon) -> Void

    func makeUIView(context: Context) -> EmojiTextView {
        let view = EmojiTextView()
        view.font = .systemFont(ofSize: 34)
        view.textAlignment = .center
        view.isScrollEnabled = false
        view.backgroundColor = .clear
        view.autocorrectionType = .no
        view.spellCheckingType = .no
        view.delegate = context.coordinator
        view.accessibilityLabel = "Emoji or Genmoji"
        view.accessibilityHint = "Opens the emoji keyboard"
        if #available(iOS 18.0, *) { view.supportsAdaptiveImageGlyph = true }
        return view
    }

    func updateUIView(_ view: EmojiTextView, context: Context) {
        context.coordinator.parent = self
    }

    func makeCoordinator() -> Coordinator { Coordinator(parent: self) }

    final class Coordinator: NSObject, UITextViewDelegate {
        var parent: GenmojiField
        init(parent: GenmojiField) { self.parent = parent }

        func textViewDidBeginEditing(_ view: UITextView) { parent.isEditing = true }
        func textViewDidEndEditing(_ view: UITextView) { parent.isEditing = false }

        func textViewDidChange(_ view: UITextView) {
            guard let icon = Self.lastIcon(in: view.attributedText) else { return }
            parent.onPick(icon)
            view.attributedText = NSAttributedString()
        }

        static func lastIcon(in text: NSAttributedString) -> ThingIcon? {
            if #available(iOS 18.0, *) {
                var glyph: NSAdaptiveImageGlyph?
                text.enumerateAttribute(.adaptiveImageGlyph, in: NSRange(location: 0, length: text.length)) { value, _, _ in
                    if let value = value as? NSAdaptiveImageGlyph { glyph = value }
                }
                if let glyph { return .genmoji(glyph.imageContent, description: glyph.contentDescription) }
            }
            return text.string.last(where: \.isEmoji).map { .emoji(String($0)) }
        }
    }
}

/// Opens straight onto the emoji keyboard when one is installed.
private final class EmojiTextView: UITextView {
    override var textInputMode: UITextInputMode? {
        UITextInputMode.activeInputModes.first { $0.primaryLanguage == "emoji" } ?? super.textInputMode
    }
}

private extension Character {
    /// Pictures, not digits or "#", which Unicode also counts as emoji.
    var isEmoji: Bool {
        guard let first = unicodeScalars.first else { return false }
        return first.properties.isEmojiPresentation || (first.properties.isEmoji && unicodeScalars.count > 1)
    }
}
