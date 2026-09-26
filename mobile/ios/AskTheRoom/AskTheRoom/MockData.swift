import Foundation

enum MockData {
    /// The sample snapshot from spec section 4, plus the bridge's naming hints: thing:9 has a
    /// weak guess (still "unnamed"), thing:11 a confident one, and Grok named thing:12.
    static let sampleSnapshotJSON = """
    {"v":1,"t":1790380000.0,"table":[90,60],"online":true,"laser":{"on":true,"target":"keys"},
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
}
