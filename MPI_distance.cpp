#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

struct DataPoint {
    double x;
    double y;
    // Add other attributes as needed
};

double calculateEuclideanDistance(const DataPoint& p1, const DataPoint& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

void generateDataPoints(std::vector<DataPoint>& dataPoints, int dataSize) {
    // Generate random data points
    std::srand(std::time(0));
    for (int i = 0; i < dataSize; ++i) {
        DataPoint point;
        point.x = std::rand() / double(RAND_MAX);  // Random x-coordinate between 0 and 1
        point.y = std::rand() / double(RAND_MAX);  // Random y-coordinate between 0 and 1
        dataPoints.push_back(point);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Generate or read the data points
    std::vector<DataPoint> dataPoints;  // All data points
    std::vector<DataPoint> localDataPoints;  // Data points assigned to the current process
    const int dataSize = 100;  // Total number of data points

    // Assign data points to each process
    int localSize = dataSize / size;
    int remaining = dataSize % size;
    int localOffset = rank * localSize;
    if (rank < remaining) {
        localSize += 1;
        localOffset += rank;
    }
    else {
        localOffset += remaining;
    }

    // Generate or read the data points
    if (rank == 0) {
        // Generate the data points on rank 0
        generateDataPoints(dataPoints, dataSize);
    }

    // Scatter the data points to each process
    MPI_Scatter(&dataPoints[0], localSize * sizeof(DataPoint), MPI_BYTE,
                &localDataPoints[0], localSize * sizeof(DataPoint), MPI_BYTE,
                0, MPI_COMM_WORLD);

    // Perform local Euclidean distance calculations
    std::vector<double> localDistances(localSize);
    DataPoint targetPoint;  // The target point for distance calculation
    // Initialize the target point with your desired values

    // Start the timer
    double startTime = MPI_Wtime();

    for (int i = 0; i < localSize; ++i) {
        // Perform Euclidean distance calculation for each local data point
        localDistances[i] = calculateEuclideanDistance(localDataPoints[i], targetPoint);
    }

    // Gather local distances from all processes
    std::vector<double> allDistances(dataSize);
    MPI_Gather(&localDistances[0], localSize, MPI_DOUBLE, &allDistances[0], localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // On rank 0, perform final calculations
    if (rank == 0) {
        // Perform final calculations (e.g., finding the minimum distance)
        // You can use the gathered distances in 'allDistances' to perform further computations

        // Stop the timer
        double endTime = MPI_Wtime();
        double totalTime = endTime - startTime;

        std::cout << "Total time: " << totalTime << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
