using Microsoft.ML.Data;

namespace Lab2.Models;

public class LiverCaseCovidPrediction
{
    //[ColumnName("PredictedLabel")]
    //public string Pandemic;

    [ColumnName("PredictedLabel")]
    public bool Pandemic;
}