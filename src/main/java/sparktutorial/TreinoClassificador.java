package sparktutorial;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.SparkSession.Builder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.when;
import static org.apache.spark.sql.functions.col;

public class TreinoClassificador {
	
	private static double HAM = 0.0, SPAM = 1.0;
	
	public static void main(String[] args) throws IOException {
		// Inicialização
		Builder builder = SparkSession.builder().appName("ClassificadorDeSpam");
		if (args.length > 1 && args[1].equals("-l")) builder.master("local[*]");
		SparkSession session = builder.getOrCreate();

		// Dados
		StructType schema = new StructType(new StructField[] {
			new StructField("label", DataTypes.StringType, false, Metadata.empty()),
			new StructField("value", DataTypes.StringType, false, Metadata.empty())
		});
		Dataset<Row> mensagens = 
				session
					.read()
					.format("csv")
					.option("delimiter", ",")
					.option("header", "true")
					.schema(schema)
					.load(args[0])
					// Tratamento dos labels
					.select(
						when(col("label").equalTo("ham"), HAM).otherwise(SPAM).as("label"),
						col("value")
					);
		
		Dataset<Row>[] splits = mensagens.randomSplit(new double[]{0.9, 0.1}, 1234);
		Dataset<Row> treino = splits[0].cache();
		Dataset<Row> teste = splits[1].cache();
		
		// Treinamento do classificador
		Tokenizer tokenizer = new Tokenizer()
				.setInputCol("value")
				.setOutputCol("palavras");
		
		Word2Vec word2Vec = new Word2Vec()
				  .setInputCol("palavras")
				  .setOutputCol("features");

		LogisticRegression lr = new LogisticRegression()
				.setMaxIter(10)
				.setRegParam(0.3)
				.setElasticNetParam(0.8);
		
		Pipeline pipeline = new Pipeline()
				  .setStages(new PipelineStage[] {tokenizer, word2Vec, lr});
		
		ParamMap[] paramGrid = new ParamGridBuilder()
				  .addGrid(word2Vec.vectorSize(), new int[] {10, 20, 30})
				  .addGrid(word2Vec.minCount(), new int[] {5, 10, 15})
				  .addGrid(lr.regParam(), new double[] {0.01, 0.1, 0.3})
				  .addGrid(lr.maxIter(), new int[] {50, 100, 300})
				  .addGrid(lr.elasticNetParam(), new double[] {0.1, 0.5, 0.8})
				  .build();
		
		CrossValidator cv = new CrossValidator()
				  .setEstimator(pipeline)
				  .setEvaluator(new BinaryClassificationEvaluator())
				  .setEstimatorParamMaps(paramGrid)
				  .setNumFolds(5);
		
		CrossValidatorModel modelo = cv.fit(treino);
		
		// Teste
		Dataset<Row> predicoes = modelo.transform(teste);
		predicoes.groupBy(col("prediction")).count().show();

		// Salvando os modelos
		File pasta = new File("data/modelo_atual");
		if (pasta.exists()) FileUtils.deleteDirectory(pasta);
		modelo.save(pasta.getAbsolutePath());
		
		session.close();
	}

}
