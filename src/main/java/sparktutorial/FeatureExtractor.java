package sparktutorial;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

@SuppressWarnings("serial")
public class FeatureExtractor implements MapFunction<Row, Row> {
	
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
			new StructField("emp.var.rate", DataTypes.DoubleType, false, Metadata.empty()),
			new StructField("cons.price.idx", DataTypes.DoubleType, false, Metadata.empty()),
			new StructField("cons.conf.idx", DataTypes.DoubleType, false, Metadata.empty()),
			new StructField("euribor3m", DataTypes.DoubleType, false, Metadata.empty()),
			new StructField("nr.employed", DataTypes.DoubleType, false, Metadata.empty()),
			new StructField("y", DataTypes.StringType, false, Metadata.empty()),
	});
	
	public static StructType SCHEMA_TREINO = new StructType(new StructField[] {
			new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
			new StructField("features", new VectorUDT(), false, Metadata.empty()),
	});

	private static List<String> job = Arrays.asList(
			"admin.","blue-collar","entrepreneur","housemaid","management","retired",
			"self-employed","services","student","technician","unemployed","unknown"
	);
	private static List<String> marital = Arrays.asList(
			"divorced","married","single","unknown"
	);
	private static List<String> education = Arrays.asList(
			"basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course",
			"university.degree","unknown"
	);
	private static List<String> defaultColumn = Arrays.asList(
			"no","yes","unknown"
	);
	private static List<String> housing = Arrays.asList( 
			"no","yes","unknown"
	);
	private static List<String> loan = Arrays.asList(
			"no","yes","unknown"
	);
	private static List<String> contact = Arrays.asList(
			"cellular","telephone"
	);
	private static List<String> month = Arrays.asList(
			"jan", "feb", "mar","apr","may","jun","jul","aug","sep","oct","nov", "dec"
	);
	private static List<String> dayOfWeek = Arrays.asList(
			"mon","tue","wed","thu","fri"
	);
	private static List<String> pOutcome = Arrays.asList(
			"failure","nonexistent","success"
	);
	private static List<String> label = Arrays.asList(
			"no","yes"
	);
	
	@Override
	public Row call(Row r) throws Exception {
		String lbl = r.<String>getAs("y");
		double rotulo = Double.valueOf(label.indexOf(lbl));
		double[] caracteristicas = new double[] {
				r.<Double>getAs("age"),
				r.<Double>getAs("campaign"),
				r.<Double>getAs("pdays"),
				r.<Double>getAs("previous"),
				r.<Double>getAs("emp.var.rate"),
				r.<Double>getAs("cons.price.idx"),
				r.<Double>getAs("cons.conf.idx"),
				r.<Double>getAs("euribor3m"),
				r.<Double>getAs("nr.employed"),
				Double.valueOf(job.indexOf(r.<String>getAs("job"))),
				Double.valueOf(marital.indexOf(r.<String>getAs("marital"))),
				Double.valueOf(education.indexOf(r.<String>getAs("education"))),
				Double.valueOf(defaultColumn.indexOf(r.<String>getAs("default"))),
				Double.valueOf(housing.indexOf(r.<String>getAs("housing"))),
				Double.valueOf(loan.indexOf(r.<String>getAs("loan"))),
				Double.valueOf(contact.indexOf(r.<String>getAs("contact"))),
				Double.valueOf(month.indexOf(r.<String>getAs("month"))),
				Double.valueOf(dayOfWeek.indexOf(r.<String>getAs("day_of_week"))),
				Double.valueOf(pOutcome.indexOf(r.<String>getAs("poutcome"))),
		};
		return RowFactory.create(rotulo, Vectors.dense(caracteristicas));
	}

}
