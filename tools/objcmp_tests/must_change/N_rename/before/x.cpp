namespace ORB_SLAM3 { struct ORBmatcher { static int TH_HIGH_score(int); }; struct ORBdescriptor { static int TH_HIGH_score(int); }; }
int use(int d) { return ORB_SLAM3::ORBmatcher::TH_HIGH_score(d); }
