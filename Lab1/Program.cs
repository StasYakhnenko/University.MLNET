using Lab1.Models;
using Microsoft.ML;

string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "train.csv");

MLContext mlContext = new MLContext();

IDataView dataView = mlContext.Data.LoadFromTextFile<SalesData>(dataPath, hasHeader: true, separatorChar: ',');

var data = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.8);

var model = Train(mlContext);

Evaluate(mlContext, model);

ITransformer Train(MLContext ctx)
{
    var pipeline =
        ctx.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(SalesData.Purchase))
            .Append(ctx.Transforms.ReplaceMissingValues(nameof(SalesData.ProductCategory2)))
            .Append(ctx.Transforms.ReplaceMissingValues(nameof(SalesData.ProductCategory3)))
            .Append(ctx.Transforms.Categorical.OneHotEncoding(outputColumnName: "GenderEncoded", inputColumnName: nameof(SalesData.Gender)))
            .Append(ctx.Transforms.Categorical.OneHotEncoding(outputColumnName: "AgeEncoded", inputColumnName: nameof(SalesData.Age)))
            .Append(ctx.Transforms.Categorical.OneHotEncoding(outputColumnName: "CityCategoryEncoded", inputColumnName: nameof(SalesData.CityCategory)))
            .Append(ctx.Transforms.Categorical.OneHotEncoding(outputColumnName: "StayInCurrentCityYearsEncoded", inputColumnName: nameof(SalesData.StayInCurrentCityYears)))
            .Append(ctx.Transforms.Concatenate("Features",
                "GenderEncoded",
                "AgeEncoded",
                nameof(SalesData.Occupation),
                "StayInCurrentCityYearsEncoded",
                "CityCategoryEncoded",
                nameof(SalesData.MaritalStatus),
                nameof(SalesData.ProductCategory1),
                nameof(SalesData.ProductCategory2),
                nameof(SalesData.ProductCategory3)
                )
            )
            .Append(ctx.Regression.Trainers.FastTree());

    return pipeline.Fit(data.TrainSet);
}

void Evaluate(MLContext ctx, ITransformer model)
{
    var predictions = model.Transform(data.TestSet);

    var metrics = ctx.Regression.Evaluate(predictions);

    Console.WriteLine();
    Console.WriteLine($"*************************************************");
    Console.WriteLine($"*       Model quality metrics evaluation         ");
    Console.WriteLine($"*------------------------------------------------");
    Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
    Console.WriteLine($"*------------------------------------------------");
    Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
    Console.WriteLine($"*************************************************");
}