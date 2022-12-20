using Microsoft.ML.Data;

namespace Lab1.Models;

/// <summary>
/// Sales data.
/// </summary>
public class SalesData
{
    [LoadColumn(2)]
    public string Gender { get; set; } = null!;

    [LoadColumn(3)]
    public string Age { get; set; } = null!;

    [LoadColumn(4)]
    public float Occupation { get; set; }

    [LoadColumn(5)]
    public string CityCategory { get; set; } = null!;

    [LoadColumn(6)]
    public string StayInCurrentCityYears { get; set; } = null!;

    [LoadColumn(7)]
    public float MaritalStatus { get; set; }

    [LoadColumn(8)]
    public float ProductCategory1 { get; set; }

    [LoadColumn(9)]
    public float ProductCategory2 { get; set; }

    [LoadColumn(10)]
    public float ProductCategory3 { get; set; }

    [LoadColumn(11)]
    public float Purchase { get; set; }
}