/**
 * Tier 2 GTest: Map container operations
 *
 * Tests Map's core bookkeeping: adding/erasing MapPoints,
 * big-change index, IMU initialization flags, and bad-map flag.
 * KeyFrame insertion is avoided because AddKeyFrame dereferences
 * pKF->mnId, which requires a fully constructed KeyFrame.
 */

#include <gtest/gtest.h>
#include "Map.hpp"
#include "MapPoint.hpp"
#include "KeyFrame.hpp"

// ---------------------------------------------------------------------------
// Fixture: creates a fresh Map(0) for each test
// ---------------------------------------------------------------------------
class MapTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Save and later restore static counters so tests are independent
        savedMapNextId = ORB_SLAM3::Map::nNextId;
        savedMPNextId  = ORB_SLAM3::MapPoint::nNextId;
    }
    void TearDown() override {
        ORB_SLAM3::Map::nNextId      = savedMapNextId;
        ORB_SLAM3::MapPoint::nNextId = savedMPNextId;
    }

    long unsigned int savedMapNextId;
    long unsigned int savedMPNextId;
};

// ---------------------------------------------------------------------------
// Constructor: initial state
// ---------------------------------------------------------------------------
TEST_F(MapTest, ConstructorInitialState) {
    ORB_SLAM3::Map testMap(0);
    EXPECT_EQ(testMap.KeyFramesInMap(), 0u);
    EXPECT_EQ(testMap.MapPointsInMap(), 0u);
    EXPECT_FALSE(testMap.IsBad());
    EXPECT_FALSE(testMap.isImuInitialized());
    EXPECT_FALSE(testMap.IsInertial());
}

TEST_F(MapTest, ConstructorSetsInitKFid) {
    ORB_SLAM3::Map testMap(42);
    EXPECT_EQ(testMap.GetInitKFid(), 42u);
}

// ---------------------------------------------------------------------------
// Map ID auto-increments via nNextId
// ---------------------------------------------------------------------------
TEST_F(MapTest, MapIdAutoIncrements) {
    ORB_SLAM3::Map::nNextId = 100;
    ORB_SLAM3::Map m1(0);
    ORB_SLAM3::Map m2(0);
    // m1 gets id 100, m2 gets id 101
    EXPECT_EQ(m1.GetId(), 100u);
    EXPECT_EQ(m2.GetId(), 101u);
}

// ---------------------------------------------------------------------------
// AddMapPoint / MapPointsInMap
// ---------------------------------------------------------------------------
TEST_F(MapTest, AddMapPointIncreasesCount) {
    ORB_SLAM3::Map testMap(0);
    ORB_SLAM3::MapPoint mp1, mp2, mp3;

    testMap.AddMapPoint(&mp1);
    EXPECT_EQ(testMap.MapPointsInMap(), 1u);

    testMap.AddMapPoint(&mp2);
    testMap.AddMapPoint(&mp3);
    EXPECT_EQ(testMap.MapPointsInMap(), 3u);
}

TEST_F(MapTest, AddSameMapPointTwiceNoDuplicate) {
    ORB_SLAM3::Map testMap(0);
    ORB_SLAM3::MapPoint mp;

    testMap.AddMapPoint(&mp);
    testMap.AddMapPoint(&mp); // std::set ignores duplicate
    EXPECT_EQ(testMap.MapPointsInMap(), 1u);
}

// ---------------------------------------------------------------------------
// EraseMapPoint / MapPointsInMap
// ---------------------------------------------------------------------------
TEST_F(MapTest, EraseMapPointDecreasesCount) {
    ORB_SLAM3::Map testMap(0);
    ORB_SLAM3::MapPoint mp1, mp2;

    testMap.AddMapPoint(&mp1);
    testMap.AddMapPoint(&mp2);
    EXPECT_EQ(testMap.MapPointsInMap(), 2u);

    testMap.EraseMapPoint(&mp1);
    EXPECT_EQ(testMap.MapPointsInMap(), 1u);
}

TEST_F(MapTest, EraseNonExistentMapPointIsHarmless) {
    ORB_SLAM3::Map testMap(0);
    ORB_SLAM3::MapPoint mp;

    // Erasing something that was never added should not crash
    testMap.EraseMapPoint(&mp);
    EXPECT_EQ(testMap.MapPointsInMap(), 0u);
}

// ---------------------------------------------------------------------------
// GetAllMapPoints returns all inserted points
// ---------------------------------------------------------------------------
TEST_F(MapTest, GetAllMapPointsReturnsInserted) {
    ORB_SLAM3::Map testMap(0);
    ORB_SLAM3::MapPoint mp1, mp2;

    testMap.AddMapPoint(&mp1);
    testMap.AddMapPoint(&mp2);

    std::vector<ORB_SLAM3::MapPoint*> allPts = testMap.GetAllMapPoints();
    EXPECT_EQ(allPts.size(), 2u);
}

// ---------------------------------------------------------------------------
// SetBad / IsBad
// ---------------------------------------------------------------------------
TEST_F(MapTest, SetBadMakesMapBad) {
    ORB_SLAM3::Map testMap(0);
    EXPECT_FALSE(testMap.IsBad());
    testMap.SetBad();
    EXPECT_TRUE(testMap.IsBad());
}

// ---------------------------------------------------------------------------
// InformNewBigChange / GetLastBigChangeIdx
// ---------------------------------------------------------------------------
TEST_F(MapTest, BigChangeIdxStartsAtZero) {
    ORB_SLAM3::Map testMap(0);
    EXPECT_EQ(testMap.GetLastBigChangeIdx(), 0);
}

TEST_F(MapTest, InformNewBigChangeIncrementsIdx) {
    ORB_SLAM3::Map testMap(0);
    testMap.InformNewBigChange();
    EXPECT_EQ(testMap.GetLastBigChangeIdx(), 1);
    testMap.InformNewBigChange();
    EXPECT_EQ(testMap.GetLastBigChangeIdx(), 2);
}

// ---------------------------------------------------------------------------
// SetImuInitialized / isImuInitialized
// ---------------------------------------------------------------------------
TEST_F(MapTest, ImuInitializedDefaultFalse) {
    ORB_SLAM3::Map testMap(0);
    EXPECT_FALSE(testMap.isImuInitialized());
}

TEST_F(MapTest, SetImuInitializedMakesItTrue) {
    ORB_SLAM3::Map testMap(0);
    testMap.SetImuInitialized();
    EXPECT_TRUE(testMap.isImuInitialized());
}

// ---------------------------------------------------------------------------
// SetInertialSensor / IsInertial
// ---------------------------------------------------------------------------
TEST_F(MapTest, InertialDefaultFalse) {
    ORB_SLAM3::Map testMap(0);
    EXPECT_FALSE(testMap.IsInertial());
}

TEST_F(MapTest, SetInertialSensorMakesItTrue) {
    ORB_SLAM3::Map testMap(0);
    testMap.SetInertialSensor();
    EXPECT_TRUE(testMap.IsInertial());
}

// ---------------------------------------------------------------------------
// SetIniertialBA1 / SetIniertialBA2  (note the typo is in the original API)
// ---------------------------------------------------------------------------
TEST_F(MapTest, InertialBADefaultFalse) {
    ORB_SLAM3::Map testMap(0);
    EXPECT_FALSE(testMap.GetIniertialBA1());
    EXPECT_FALSE(testMap.GetIniertialBA2());
}

TEST_F(MapTest, SetIniertialBA1And2) {
    ORB_SLAM3::Map testMap(0);
    testMap.SetIniertialBA1();
    EXPECT_TRUE(testMap.GetIniertialBA1());
    EXPECT_FALSE(testMap.GetIniertialBA2());

    testMap.SetIniertialBA2();
    EXPECT_TRUE(testMap.GetIniertialBA2());
}

// ---------------------------------------------------------------------------
// ChangeIndex tracking
// ---------------------------------------------------------------------------
TEST_F(MapTest, MapChangeIndexStartsAtZero) {
    ORB_SLAM3::Map testMap(0);
    EXPECT_EQ(testMap.GetMapChangeIndex(), 0);
}

TEST_F(MapTest, IncreaseChangeIndexIncrements) {
    ORB_SLAM3::Map testMap(0);
    testMap.IncreaseChangeIndex();
    EXPECT_EQ(testMap.GetMapChangeIndex(), 1);
    testMap.IncreaseChangeIndex();
    EXPECT_EQ(testMap.GetMapChangeIndex(), 2);
}

// ---------------------------------------------------------------------------
// SetCurrentMap / IsInUse
// ---------------------------------------------------------------------------
TEST_F(MapTest, SetCurrentMapMarksInUse) {
    ORB_SLAM3::Map testMap(0);
    EXPECT_FALSE(testMap.IsInUse());
    testMap.SetCurrentMap();
    EXPECT_TRUE(testMap.IsInUse());
}

TEST_F(MapTest, SetStoredMapClearsInUse) {
    ORB_SLAM3::Map testMap(0);
    testMap.SetCurrentMap();
    EXPECT_TRUE(testMap.IsInUse());
    testMap.SetStoredMap();
    EXPECT_FALSE(testMap.IsInUse());
}
