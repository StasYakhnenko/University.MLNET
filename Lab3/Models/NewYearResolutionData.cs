using Microsoft.ML.Data;

namespace Lab3.Models;

/// <summary>
/// New year resolution data.
/// </summary>
public class NewYearResolutionData
{
    /// <summary>
    /// Resolution topics.
    /// Example: 'Eat healthier', 'Be more positive'.
    /// </summary>
    [LoadColumn(0)]
    public string ResolutionTopics { get; set; } = null!;

    /// <summary>
    /// Resolution category.
    /// Example: 'Health & Fitness', 'Personal growth'.
    /// </summary>
    [LoadColumn(3)]
    public string ResolutionCategory { get; set; } = null!;

    /// <summary>
    /// Text of the resolution.
    /// </summary>
    [LoadColumn(5)]
    public string Text { get; set; } = null!;
}