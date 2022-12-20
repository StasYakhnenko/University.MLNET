using Microsoft.ML;
using System.Data;

namespace Lab4.Helpers;

public static class DataViewHelper
{
    public static DataTable ToDataTable(this IDataView dataView)
    {
        var dt = new DataTable();
        var preview = dataView.Preview(int.MaxValue);
        dt.Columns.AddRange(preview.Schema.Select(x => new DataColumn(x.Name)).ToArray());
        foreach (var row in preview.RowView)
        {
            var r = dt.NewRow();
            foreach (var col in row.Values)
            {
                r[col.Key] = col.Value;
            }
            dt.Rows.Add(r);
        }

        return dt;
    }
}