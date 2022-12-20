using Microsoft.ML.Data;

namespace Lab1.Models;

public class SalesPrediction
{
    [ColumnName("Score")]
    public float Purchase { get; set; }
}