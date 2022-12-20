using Microsoft.ML.Data;

namespace Lab2.Models;

/// <summary>
/// Liver cancer case.
/// </summary>
public class LiverCancerCase
{
    [LoadColumn(0)]
    public string Cancer { get; set; } = null!;

    [LoadColumn(1)]
    public string Year { get; set; } = null!;
    //[LoadColumn(2)]
    //public float Month { get; set; }
    [LoadColumn(3)]
    public string Bleed { get; set; } = null!;
    [LoadColumn(4)]
    public string ModePresentation { get; set; } = null!;
    //[LoadColumn(5)]
    //public float Age { get; set; }
    [LoadColumn(6)]
    public string Gender { get; set; } = null!;
    [LoadColumn(7)]
    public string Etiology { get; set; } = null!;
    [LoadColumn(8)]
    public string Cirrhois { get; set; } = null!;
    [LoadColumn(9)]
    public string Size { get; set; }
    [LoadColumn(10)]
    public string HccTnmStage { get; set; } = null!;
    [LoadColumn(11)]
    public string HccBclcStage { get; set; } = null!;
    [LoadColumn(12)]
    public string IccTnmStage { get; set; } = null!;
    [LoadColumn(13)]
    public string TreatmentGrps { get; set; } = null!;
    [LoadColumn(14)]
    public string SurvivalFromMdm { get; set; } = null!;
    [LoadColumn(15)]
    public string AliveDead { get; set; } = null!;
    [LoadColumn(16)]
    public string TypeOfIncidentalFinding { get; set; } = null!;
    [LoadColumn(17)]
    public string SurveillanceProgramme { get; set; } = null!;
    [LoadColumn(18)]
    public string SurveillanceEffectiveness { get; set; } = null!;
    [LoadColumn(19)]
    public string ModeOfSurveillanceDetection { get; set; } = null!;
    [LoadColumn(20)]
    public string TimeDiagnosis1StTx { get; set; } = null!;
    [LoadColumn(21)]
    public string DateIncidentSurveillanceScan { get; set; } = null!;
    [LoadColumn(22)]
    public string Ps { get; set; } = null!;
    [LoadColumn(23)]
    public string TimeMdm1StTreatment { get; set; } = null!;
    [LoadColumn(24)]
    public string TimeDecisionToTreat1StTreatment { get; set; } = null!;
    [LoadColumn(25)]
    public string PrevKnownCirrhosis { get; set; } = null!;
    [LoadColumn(26)]
    public string MonthsFromLastSurveillance { get; set; }
}