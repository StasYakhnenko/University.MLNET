using Lab2.Models;
using Microsoft.ML;

var dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "covid-liver.csv");

var mlContext = new MLContext();
var dataView = mlContext.Data.LoadFromTextFile<LiverCancerCase>(dataPath, hasHeader: true, separatorChar:',');
var data = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.4);

var pipeline = ProcessData();
ITransformer trainedModel = BuildAndTrainModel(pipeline);
Evaluate();

ITransformer BuildAndTrainModel(IEstimator<ITransformer> pipeline)
{
    var trainingPipeline = pipeline
        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

    return trainingPipeline.Fit(data.TrainSet);
}

IEstimator<ITransformer> ProcessData()
{
    var targetMap = new Dictionary<string, bool> { { "\"Prepandemic\"", false }, { "\"Pandemic\"", true } };

    return mlContext.Transforms.Conversion.MapValue("Label", targetMap, inputColumnName: "Year")
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "CancerEncoded",
            inputColumnName: nameof(LiverCancerCase.Cancer)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "BleedEncoded",
            inputColumnName: nameof(LiverCancerCase.Bleed)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ModePresentationEncoded",
            inputColumnName: nameof(LiverCancerCase.ModePresentation)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "GenderEncoded",
            inputColumnName: nameof(LiverCancerCase.Gender)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "EtiologyEncoded",
            inputColumnName: nameof(LiverCancerCase.Etiology)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "CirrhoisEncoded",
            inputColumnName: nameof(LiverCancerCase.Cirrhois)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "HccTnmStageEncoded",
            inputColumnName: nameof(LiverCancerCase.HccTnmStage)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "HccBclcStageEncoded",
            inputColumnName: nameof(LiverCancerCase.HccBclcStage)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "IccTnmStageEncoded",
            inputColumnName: nameof(LiverCancerCase.IccTnmStage)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "TreatmentGrpsEncoded",
            inputColumnName: nameof(LiverCancerCase.TreatmentGrps)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "SurvivalFromMdmEncoded",
            inputColumnName: nameof(LiverCancerCase.SurvivalFromMdm)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "AliveDeadEncoded",
            inputColumnName: nameof(LiverCancerCase.AliveDead)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "TypeOfIncidentalFindingEncoded",
            inputColumnName: nameof(LiverCancerCase.TypeOfIncidentalFinding)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "SurveillanceProgrammeEncoded",
            inputColumnName: nameof(LiverCancerCase.SurveillanceProgramme)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "SurveillanceEffectivenessEncoded",
            inputColumnName: nameof(LiverCancerCase.SurveillanceEffectiveness)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ModeOfSurveillanceDetectionEncoded",
            inputColumnName: nameof(LiverCancerCase.ModeOfSurveillanceDetection)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "TimeDiagnosis1StTxEncoded",
            inputColumnName: nameof(LiverCancerCase.TimeDiagnosis1StTx)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(
            outputColumnName: "DateIncidentSurveillanceScanEncoded",
            inputColumnName: nameof(LiverCancerCase.DateIncidentSurveillanceScan)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PsEncoded",
            inputColumnName: nameof(LiverCancerCase.Ps)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "TimeMdm1StTreatmentEncoded",
            inputColumnName: nameof(LiverCancerCase.TimeMdm1StTreatment)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(
            outputColumnName: "TimeDecisionToTreat1StTreatmentEncoded",
            inputColumnName: nameof(LiverCancerCase.TimeDecisionToTreat1StTreatment)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PrevKnownCirrhosisEncoded",
            inputColumnName: nameof(LiverCancerCase.PrevKnownCirrhosis)))
        .Append(mlContext.Transforms.Concatenate("Features",
                "CancerEncoded",
                "BleedEncoded",
                "ModePresentationEncoded",
                "GenderEncoded",
                "EtiologyEncoded",
                "CirrhoisEncoded",
                "HccTnmStageEncoded",
                "HccBclcStageEncoded",
                "IccTnmStageEncoded",
                "TreatmentGrpsEncoded",
                "SurvivalFromMdmEncoded",
                "AliveDeadEncoded",
                "TypeOfIncidentalFindingEncoded",
                "SurveillanceProgrammeEncoded",
                "SurveillanceEffectivenessEncoded",
                "ModeOfSurveillanceDetectionEncoded",
                "TimeDiagnosis1StTxEncoded",
                "PsEncoded",
                "PrevKnownCirrhosisEncoded"
            )
        );
}

void Evaluate()
{
    var testDataView = data.TrainSet;

    var testMetrics = mlContext.BinaryClassification.Evaluate(trainedModel.Transform(testDataView));

    Console.WriteLine($"*************************************************************************************************************");
    Console.WriteLine($"*       Metrics for Binary Classification model - Test Data     ");
    Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
    Console.WriteLine($"*       Accuracy:         {testMetrics.Accuracy:0.###}");
    Console.WriteLine($"*       F1:               {testMetrics.F1Score:0.###}");
    Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
    Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
    Console.WriteLine($"*************************************************************************************************************");
}