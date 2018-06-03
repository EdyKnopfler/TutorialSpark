package sparktutorial;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.SparkSession.Builder;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;

import static org.apache.spark.sql.functions.col;

public class TreinoClassificador {
	
	public static void main(String[] args) throws IOException {
		// Inicialização
		Builder builder = SparkSession.builder().appName("ClassificadorBancario");
		if (args.length > 1 && args[1].equals("-l")) builder.master("local[*]");
		SparkSession session = builder.getOrCreate();
		Logger.getRootLogger().setLevel(Level.WARN);

		// Dados
		Dataset<Row> dadosClientes = session
				.read()
				.format("csv")
				.option("delimiter", ";")
				.option("header", "true")
				.schema(FeatureExtractor.SCHEMA_DADOS)
				.load(args[0])
				.drop("duration");  // Esta deve ser descartada
		
		Dataset<Row> dataset = dadosClientes.map(new FeatureExtractor(), 
				RowEncoder.apply(FeatureExtractor.SCHEMA_TREINO));
		
		// Tratamento
		StandardScaler escalador = new StandardScaler()
				  .setInputCol("features")
				  .setOutputCol("scaledFeatures")
				  .setWithStd(true)
				  .setWithMean(false);

		StandardScalerModel escala = escalador.fit(dataset);
		Dataset<Row> dadosEscalados = escala.transform(dataset);
		
		// Treinamento do classificador
		Dataset<Row>[] splits = dadosEscalados.randomSplit(new double[]{0.8, 0.2});
		Dataset<Row> treino = splits[0].cache();
		Dataset<Row> teste = splits[1].cache();
		
		LogisticRegression classificador = new LogisticRegression()
				  .setMaxIter(10)
				  .setRegParam(0.3)
				  .setElasticNetParam(0.8);

		ParamMap[] paramGrid = new ParamGridBuilder()
				.addGrid(classificador.maxIter(), new int[] {10, 30, 50})
				.addGrid(classificador.regParam(), new double[] {0.01, 0.1, 0.3})
				.addGrid(classificador.elasticNetParam(), new double[] {0.9, 0.9, 0.99})
				.build();

		BinaryClassificationEvaluator avaliador = new BinaryClassificationEvaluator();
		
		TrainValidationSplit tvs = new TrainValidationSplit()
				  .setEstimator(classificador)
				  .setEvaluator(avaliador)
				  .setEstimatorParamMaps(paramGrid)
				  .setTrainRatio(0.8);
		
		TrainValidationSplitModel modelo = tvs.fit(treino);
		
		// Teste
		Dataset<Row> predicoes = modelo.transform(teste);
		System.out.println("Acurácia: " + avaliador.evaluate(predicoes));
		predicoes.groupBy(col("prediction"), col("label")).count().show();

		// Salvando os modelos
		File pasta = new File("data/modelo_atual");
		if (pasta.exists()) FileUtils.deleteDirectory(pasta);
		modelo.save(pasta.getAbsolutePath());
		
		session.close();
	}

}
