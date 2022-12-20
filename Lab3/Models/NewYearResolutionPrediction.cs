using Microsoft.ML.Data;

namespace Lab3.Models;

/// <summary>
/// New year resolution prediction.
/// </summary>
public class NewYearResolutionPrediction
{
    [ColumnName("PredictedLabel")]
    public uint PredictedClusterId { get; set; }

    [ColumnName("Score")]
    public float[] Distances { get; set; } = null!;
}