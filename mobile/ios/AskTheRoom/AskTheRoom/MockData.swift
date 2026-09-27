import Foundation

enum MockData {
    /// The sample snapshot from spec section 4, plus the bridge's naming hints: thing:9 has a
    /// weak guess (still "something new"), thing:11 a confident one, and Grok named thing:12. The map is
    /// turned for someone on the couch, the camera's right side, so the "You" marker shows.
    static let sampleSnapshotJSON = """
    {"v":1,"t":1790380000.0,"table":[90,60],"online":true,"laser":{"on":true,"target":"keys"},
     "view":{"f":"right","o":true,"s":{"right":"couch"}},
     "e":[
      {"n":"keys","k":"t","s":"I","p":"box","xy":[41.2,29.0],"r":[70.4,38.1],"c":0.85,"ls":1790379880.0},
      {"n":"box","k":"c","s":"V","xy":[70.4,38.1],"r":[70.4,38.1],"c":1.0},
      {"n":"pill_bottle","k":"t","s":"U","p":"notebook","xy":[20.0,40.0],"r":[20.5,39.0],"c":0.85},
      {"n":"notebook","k":"v","s":"V","xy":[20.5,39.0],"r":[20.5,39.0],"c":1.0},
      {"n":"wallet","k":"t","s":"V","xy":[60.0,15.0],"r":[60.0,15.0],"c":1.0},
      {"n":"phone","k":"t","s":"G","xy":[3.0,30.0],"c":0.9,"edge":"left"},
      {"n":"glasses","k":"t","s":"X","xy":[80.0,50.0],"c":0.4},
      {"n":"remote","k":"t","s":"H","p":"hand:2","xy":[50.0,45.0],"r":[50.0,45.0],"c":0.9},
      {"n":"thing:7","k":"t","s":"V","xy":[33.0,12.0],"r":[33.0,12.0],"c":1.0,"a":["my charger"]},
      {"n":"thing:9","k":"t","s":"V","xy":[82.0,10.0],"r":[82.0,10.0],"c":0.6,"m":[["thing:4",0.62]],"g":"cup","gc":0.3},
      {"n":"thing:11","k":"t","s":"V","xy":[45.0,24.0],"r":[45.0,24.0],"c":1.0,"g":"phone charger","gc":0.8},
      {"n":"thing:12","k":"t","s":"V","xy":[14.0,6.0],"r":[14.0,6.0],"c":1.0,"a":["tape roll"],"as":"grok"}
     ]}
    """

    static var sampleSnapshot: Snapshot {
        // The literal above is fixed and covered by tests, so this can't fail at runtime.
        Wire.decode(Snapshot.self, from: Data(sampleSnapshotJSON.utf8))!
    }

    /// The real room seen from the couch (the rig turns it to the seat, front = camera right):
    /// the couch along the near side with you on it, the TV stand (camera left, past the far
    /// edge) across the far wall, the doorway (far back) on the right wall, and the counter
    /// (camera far right) in the near right corner. Centimetres, schematic. The table is the
    /// sample's 90 x 60, and `origin` is its top-left, so a table thing sits at origin + r.
    static let sampleLayoutJSON = """
    {"v":1,"size":[320,280],"front":"right",
     "table":{"rect":[105,100,90,60],"origin":[105,100]},
     "zones":[
      {"id":"side_table","say":"the TV stand","rect":[92,8,136,38],"kind":"surface"},
      {"id":"doorway","say":"the doorway","rect":[276,20,36,104],"kind":"door"},
      {"id":"counter","say":"the counter","rect":[246,170,66,102],"kind":"surface"},
      {"id":"couch","say":"the couch","rect":[24,198,206,74],"kind":"seat"}
     ],
     "you":[184,254]}
    """

    static var sampleLayout: RoomLayout {
        Wire.decode(RoomLayout.self, from: Data(sampleLayoutJSON.utf8))!
    }

    /// Things around the room rather than on the table, one of each look: seen on the couch,
    /// carried off from the counter, hidden in the TV stand, and last seen by the door.
    static let roomEntitiesJSON = """
    [
     {"n":"headphones","k":"t","s":"V","z":"couch","c":1.0,"rg":"visible"},
     {"n":"mug","k":"t","s":"H","z":"counter","c":0.9,"rg":"carried"},
     {"n":"book","k":"t","s":"U","z":"side_table","c":0.9,"rg":"hidden"},
     {"n":"umbrella","k":"t","s":"X","z":"doorway","c":0.5,"rg":"last_seen","rt":1}
    ]
    """

    static var roomEntities: [Entity] {
        Wire.decode([Entity].self, from: Data(roomEntitiesJSON.utf8))!
    }

    /// The sample with the room around it: the layout, its hash and the room's things.
    static var roomSnapshot: Snapshot {
        var s = sampleSnapshot
        s.e += roomEntities
        s.lay = sampleLayout
        s.lh = "mock-room"
        return s
    }
}
