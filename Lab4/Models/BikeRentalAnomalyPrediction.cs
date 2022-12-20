using Microsoft.ML.Data;

namespace Lab4.Models;

/// <summary>
/// Bike rental anomaly prediction.
/// </summary>
public class BikeRentalAnomalyPrediction
{
    /// <summary>
    /// Timestamp.
    /// </summary>
    public string Timestamp { get; set; } = null!;

    /// <summary>
    /// Count.
    /// </summary>
    public float Count { get; set; }

    //vector to hold anomaly detection results. Including isAnomaly, anomalyScore, magnitude, expectedValue, boundaryUnits, upperBoundary and lowerBoundary.
    [VectorType(7)]
    public double[] Prediction { get; set; } = new double[7];
}