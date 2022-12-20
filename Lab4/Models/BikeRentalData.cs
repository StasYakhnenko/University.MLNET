using Microsoft.ML.Data;

namespace Lab4.Models;

/// <summary>
/// Bike rental data.
/// </summary>
public class BikeRentalData
{
    /// <summary>
    /// Timestamp
    /// </summary>
    [LoadColumn(1)]
    public string Timestamp { get; set; } = null!;

    /// <summary>
    /// Number of bike rentals at <see cref="Timestamp"/>.
    /// </summary>
    [LoadColumn(15)]
    public float Count { get; set; }
}