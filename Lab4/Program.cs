using Lab4.Models;
using Microsoft.ML;
using System.Drawing;
using Lab4.Helpers;
using System.Globalization;

const string dataFile = "bike_sharing_daily.csv";
const string dateFormat = "yyyy-MM-dd";

var dataPath = Path.Combine(Environment.CurrentDirectory, "Data", dataFile);

var mlContext = new MLContext();
var dataView = mlContext.Data.LoadFromTextFile<BikeRentalData>(path: dataPath, hasHeader: true, separatorChar: ',');

const int docSize = 732;

var predictions = DetectSpike(mlContext, docSize, dataView);

PrintResults(predictions);

ShowPlot(dataView, predictions);

IDataView CreateEmptyDataView(MLContext ctx) 
    => ctx.Data.LoadFromEnumerable(new List<BikeRentalData>());

BikeRentalAnomalyPrediction[] DetectSpike(MLContext ctx, int size, IDataView bikeRentals)
{
    var iidSpikeEstimator = ctx.Transforms.DetectIidSpike(
        outputColumnName: nameof(BikeRentalAnomalyPrediction.Prediction),
        inputColumnName: nameof(BikeRentalData.Count),
        confidence: (double)90,
        pvalueHistoryLength: 10);

    ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(ctx));

    IDataView transformedData = iidSpikeTransform.Transform(bikeRentals);

    return ctx.Data.CreateEnumerable<BikeRentalAnomalyPrediction>(transformedData, reuseRowObject: false).ToArray();
}

void ShowPlot(IDataView originalDataView, BikeRentalAnomalyPrediction[] spikedData)
{
    var originalData = originalDataView.ToDataTable();

    var rows = originalData.Select();

    var plt = new ScottPlot.Plot();

    // original data
    plt.AddScatter(
        rows.Select(x => (string)x["Timestamp"])
            .Select(x => DateTime.ParseExact(x, dateFormat, CultureInfo.InvariantCulture)).Select(x => x.ToOADate())
            .ToArray(),
        rows.Select(x => (string)x["Count"]).Select(Convert.ToDouble).ToArray()
    );

    var spikes = spikedData.Where(x => x.Prediction[0] == 1).ToArray();

    // spike points.
    plt.AddScatterPoints(
        spikes.Select(x => DateTime.ParseExact(x.Timestamp, dateFormat, CultureInfo.InvariantCulture))
            .Select(x => x.ToOADate()).ToArray(),
        spikes.Select(x => (double)x.Count).ToArray(),
        Color.OrangeRed,
        label: "Anomaly");

    plt.Legend();
    plt.XAxis.DateTimeFormat(true);
    new ScottPlot.FormsPlotViewer(plt).ShowDialog();
}

void PrintResults(IEnumerable<BikeRentalAnomalyPrediction> bikeRentalAnomalyPredictions)
{
    Console.WriteLine("Alert\tScore\tP-Value");

    foreach (var p in bikeRentalAnomalyPredictions)
    {
        var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";

        if (p.Prediction[0] == 1)
        {
            results += " <-- Spike detected";
        }

        Console.WriteLine(results);
    }

    Console.WriteLine("");
}