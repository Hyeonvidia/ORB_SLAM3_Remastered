namespace ORB_SLAM3 { void Relocalize(); void TrackReferenceKeyFrame(); }
void step(bool lost) { if (lost) ORB_SLAM3::Relocalize(); else ORB_SLAM3::TrackReferenceKeyFrame(); }
