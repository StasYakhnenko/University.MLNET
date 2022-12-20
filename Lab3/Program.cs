using Lab3.Models;
using Microsoft.ML;
using Microsoft.ML.Data;

var dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "new_year_resolutions_dataset.csv");
var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "new_year_resolutions_model.zip");

const int numberOfClusters = 10;

var mlContext = new MLContext(seed: 0);

IDataView dataView = mlContext.Data.LoadFromTextFile<NewYearResolutionData>(dataPath, hasHeader: true, separatorChar: ';');

var data = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.5);

string featuresColumnName = "Features";

var pipeline =
    mlContext.Transforms.Text.NormalizeText(nameof(NewYearResolutionData.Text))
        .Append(mlContext.Transforms.Text.FeaturizeText(
            outputColumnName: featuresColumnName,
            inputColumnName: nameof(NewYearResolutionData.Text)))
    .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: numberOfClusters));

var model = pipeline.Fit(data.TrainSet);

var predictions = model.Transform(data.TestSet);

var metrics = mlContext.Clustering.Evaluate(predictions, scoreColumnName: "Score", featureColumnName: "Features");

PrintMetrics(metrics);

using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
{
    mlContext.Model.Save(model, dataView.Schema, fileStream);
}

void PrintMetrics(ClusteringMetrics clusteringMetrics)
{
    Console.WriteLine();
    Console.WriteLine($"*************************************************");
    Console.WriteLine($"*       Model quality metrics evaluation         ");
    Console.WriteLine($"*------------------------------------------------");
    Console.WriteLine($"*       Number of clusters:      {numberOfClusters}");
    Console.WriteLine($"*------------------------------------------------");
    Console.WriteLine($"*       Average distance:        {clusteringMetrics.AverageDistance:#.##}");
    Console.WriteLine($"*------------------------------------------------");
    Console.WriteLine($"*       Davies Bouldin Index:    {clusteringMetrics.DaviesBouldinIndex:#.##}");
    Console.WriteLine($"*************************************************");
}