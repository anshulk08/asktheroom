// Draws the app icon: a table seen from above, a box on it, and the laser-red dot on the box.
//   swift tools/make_icon.swift AskTheRoom/AskTheRoom/Assets.xcassets/AppIcon.appiconset/AppIcon.png
import AppKit

let size = 1024
let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size, bitsPerSample: 8,
                           samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                           bytesPerRow: 0, bitsPerPixel: 0)!
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

NSColor(red: 0.17, green: 0.20, blue: 0.24, alpha: 1).setFill()
NSRect(x: 0, y: 0, width: size, height: size).fill()

// Table top, 3:2 like the real table.
NSColor(red: 0.95, green: 0.91, blue: 0.84, alpha: 1).setFill()
NSBezierPath(roundedRect: NSRect(x: 172, y: 285, width: 680, height: 454), xRadius: 56, yRadius: 56).fill()

let box = NSBezierPath(roundedRect: NSRect(x: 520, y: 380, width: 220, height: 170), xRadius: 20, yRadius: 20)
NSColor(red: 0.82, green: 0.74, blue: 0.62, alpha: 1).setFill()
box.fill()
NSColor(red: 0.45, green: 0.39, blue: 0.31, alpha: 1).setStroke()
box.lineWidth = 10
box.stroke()

let red = NSColor(red: 1, green: 0.231, blue: 0.188, alpha: 1)
let ring = NSBezierPath(ovalIn: NSRect(x: 535, y: 370, width: 190, height: 190))
red.setStroke()
ring.lineWidth = 18
ring.stroke()
red.setFill()
NSBezierPath(ovalIn: NSRect(x: 592, y: 427, width: 76, height: 76)).fill()

NSGraphicsContext.current = nil
// App icons must not have an alpha channel, so flatten to RGB.
let cg = rep.cgImage!
let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
                    space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
ctx.draw(cg, in: CGRect(x: 0, y: 0, width: size, height: size))
let out = NSBitmapImageRep(cgImage: ctx.makeImage()!)
try! out.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
