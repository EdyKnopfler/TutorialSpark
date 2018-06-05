package sparktutorial;

import org.apache.spark.sql.types.*;

public class Dados {
	
	public static StructType SCHEMA_DADOS = new StructType(new StructField[] {
			new StructField("age", DataTypes.DoubleType, false, Metadata.empty()),
			new StructField("job", DataTypes.StringType, false, Metadata.empty()),
			new StructField("marital", DataTypes.StringType, false, Metadata.empty()),
			new StructField("education", DataTypes.StringType, false, Metadata.empty()),
			new StructField("default", DataTypes.StringType, false, Metadata.empty()),
			new StructField("housing", DataTypes.StringType, false, Metadata.empty()),
			new StructField("loan", DataTypes.StringType, false, Metadata.empty()),
			new StructField("contact", DataTypes.StringType, false, Metadata.empty()),
			new StructField("month", DataTypes.StringType, false, Metadata.empty()),
			new StructField("day_of_week", DataTypes.StringType, false, Metadata.empty()),
			new StructField("duration", DataTypes.StringType, false, Metadata.empty()),
			new StructField("campaign", DataTypes.DoubleType, false, Metadata.empty()),
			new StructField("pdays", DataTypes.DoubleType, false, Metadata.empty()),
			new StructField("previous", DataTypes.DoubleType, false, Metadata.empty()),
			new StructField("poutcome", DataTypes.StringType, false, Metadata.empty()),
			new StructField("emp_var_rate", DataTypes.DoubleType, false, Metadata.empty()),
			new StructField("cons_price_idx", DataTypes.DoubleType, false, Metadata.empty()),
			new StructField("cons_conf_idx", DataTypes.DoubleType, false, Metadata.empty()),
			new StructField("euribor3m", DataTypes.DoubleType, false, Metadata.empty()),
			new StructField("nr_employed", DataTypes.DoubleType, false, Metadata.empty()),
			new StructField("y", DataTypes.StringType, false, Metadata.empty()),
	});
	
}
