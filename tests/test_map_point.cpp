/**
 * Tier 2 GTest: MapPoint basic operations
 *
 * Uses the default MapPoint constructor for most tests to avoid
 * the need for a fully constructed KeyFrame. The default constructor
 * initializes mnVisible=1, mnFound=1, mbBad=false, mpReplaced=nullptr
 * but does NOT set mnId via nNextId++ (only the parameterized
 * constructors do that). Tests that require mpMap (e.g., SetBadFlag)
 * use a real Map object with a parameterized MapPoint where possible.
 */

#include <gtest/gtest.h>
#include "MapPoint.hpp"
#include "Map.hpp"
#include "KeyFrame.hpp"

#include <Eigen/Core>

// ---------------------------------------------------------------------------
// Fixture: provides a default-constructed MapPoint for lightweight tests
// ---------------------------------------------------------------------------
class MapPointDefaultTest : public ::testing::Test {
protected:
    ORB_SLAM3::MapPoint mp;
};

// ---------------------------------------------------------------------------
// SetWorldPos / GetWorldPos roundtrip
// ---------------------------------------------------------------------------
TEST_F(MapPointDefaultTest, SetGetWorldPosRoundtrip) {
    Eigen::Vector3f pos(1.0f, 2.0f, 3.0f);
    mp.SetWorldPos(pos);
    Eigen::Vector3f retrieved = mp.GetWorldPos();
    EXPECT_FLOAT_EQ(retrieved.x(), 1.0f);
    EXPECT_FLOAT_EQ(retrieved.y(), 2.0f);
    EXPECT_FLOAT_EQ(retrieved.z(), 3.0f);
}

TEST_F(MapPointDefaultTest, SetWorldPosOverwrite) {
    Eigen::Vector3f pos1(1.0f, 0.0f, 0.0f);
    Eigen::Vector3f pos2(0.0f, 5.0f, -3.0f);
    mp.SetWorldPos(pos1);
    mp.SetWorldPos(pos2);
    Eigen::Vector3f retrieved = mp.GetWorldPos();
    EXPECT_FLOAT_EQ(retrieved.x(), 0.0f);
    EXPECT_FLOAT_EQ(retrieved.y(), 5.0f);
    EXPECT_FLOAT_EQ(retrieved.z(), -3.0f);
}

// ---------------------------------------------------------------------------
// isBad — default-constructed MapPoint is not bad
// ---------------------------------------------------------------------------
TEST_F(MapPointDefaultTest, DefaultIsNotBad) {
    EXPECT_FALSE(mp.isBad());
}

// ---------------------------------------------------------------------------
// GetReplaced — default is nullptr
// ---------------------------------------------------------------------------
TEST_F(MapPointDefaultTest, GetReplacedDefaultIsNull) {
    EXPECT_EQ(mp.GetReplaced(), nullptr);
}

// ---------------------------------------------------------------------------
// IncreaseVisible / IncreaseFound / GetFoundRatio
// Default constructor sets mnVisible=1, mnFound=1 => ratio = 1.0
// ---------------------------------------------------------------------------
TEST_F(MapPointDefaultTest, InitialFoundRatio) {
    // mnFound=1, mnVisible=1
    EXPECT_FLOAT_EQ(mp.GetFoundRatio(), 1.0f);
}

TEST_F(MapPointDefaultTest, IncreaseVisibleDecreasesRatio) {
    // mnFound=1, mnVisible=1 initially
    mp.IncreaseVisible(9); // mnVisible becomes 10
    // ratio = 1/10 = 0.1
    EXPECT_NEAR(mp.GetFoundRatio(), 0.1f, 1e-6f);
}

TEST_F(MapPointDefaultTest, IncreaseFoundIncreasesRatio) {
    mp.IncreaseVisible(9);  // mnVisible = 10
    mp.IncreaseFound(4);    // mnFound = 5
    // ratio = 5/10 = 0.5
    EXPECT_NEAR(mp.GetFoundRatio(), 0.5f, 1e-6f);
}

TEST_F(MapPointDefaultTest, GetFoundReturnsCount) {
    EXPECT_EQ(mp.GetFound(), 1); // default
    mp.IncreaseFound(3);
    EXPECT_EQ(mp.GetFound(), 4);
}

// ---------------------------------------------------------------------------
// Observations — default-constructed MapPoint has 0 observations
// ---------------------------------------------------------------------------
TEST_F(MapPointDefaultTest, DefaultObservationsIsZero) {
    EXPECT_EQ(mp.Observations(), 0);
}

TEST_F(MapPointDefaultTest, DefaultObservationsMapIsEmpty) {
    auto obs = mp.GetObservations();
    EXPECT_TRUE(obs.empty());
}

// ---------------------------------------------------------------------------
// Parameterized constructor tests (require a Map object)
// Test that mnId increments across instances.
// The (Pos, pRefKF, pMap) constructor dereferences pRefKF, so we use
// a Map and test the id mechanism through nNextId directly.
// ---------------------------------------------------------------------------
class MapPointIdTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset the static counter so test results are deterministic
        savedNextId = ORB_SLAM3::MapPoint::nNextId;
    }
    void TearDown() override {
        // Restore to avoid polluting other tests
        ORB_SLAM3::MapPoint::nNextId = savedNextId;
    }
    long unsigned int savedNextId;
};

TEST_F(MapPointIdTest, NextIdIsStatic) {
    // nNextId is a public static — verify it exists and is readable
    long unsigned int id = ORB_SLAM3::MapPoint::nNextId;
    EXPECT_GE(id, 0u);
}

TEST_F(MapPointIdTest, DefaultConstructorDoesNotIncrementId) {
    long unsigned int before = ORB_SLAM3::MapPoint::nNextId;
    ORB_SLAM3::MapPoint mp1;
    ORB_SLAM3::MapPoint mp2;
    long unsigned int after = ORB_SLAM3::MapPoint::nNextId;
    // Default constructor does not touch nNextId
    EXPECT_EQ(before, after);
}

// ---------------------------------------------------------------------------
// SetNormalVector / GetNormal roundtrip
// ---------------------------------------------------------------------------
TEST_F(MapPointDefaultTest, SetNormalVectorRoundtrip) {
    Eigen::Vector3f normal(0.0f, 1.0f, 0.0f);
    mp.SetNormalVector(normal);
    Eigen::Vector3f retrieved = mp.GetNormal();
    EXPECT_FLOAT_EQ(retrieved.x(), 0.0f);
    EXPECT_FLOAT_EQ(retrieved.y(), 1.0f);
    EXPECT_FLOAT_EQ(retrieved.z(), 0.0f);
}

// ---------------------------------------------------------------------------
// GetMap / UpdateMap
// ---------------------------------------------------------------------------
TEST(MapPointMapAssociation, UpdateMapChangesMap) {
    ORB_SLAM3::Map map1(0);
    ORB_SLAM3::Map map2(1);
    ORB_SLAM3::MapPoint mp;

    mp.UpdateMap(&map1);
    EXPECT_EQ(mp.GetMap(), &map1);

    mp.UpdateMap(&map2);
    EXPECT_EQ(mp.GetMap(), &map2);
}
