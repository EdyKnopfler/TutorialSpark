package sparktutorial;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.SparkSession.Builder;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.when;

public class TreinoClassificador {
	
	public static void main(String[] args) throws IOException {
		// Inicialização e carregamento
		Builder builder = SparkSession.builder().appName("ClassificadorBancario");
		if (args.length > 0 && args[0].equals("-l")) builder.master("local[*]");
		SparkSession session = builder.getOrCreate();
		Logger.getRootLogger().setLevel(Level.WARN);

		Dataset<Row> dadosTratados = session
				.read()
				.parquet("data/tratados")
				.cache();

		// Amostragem
		Dataset<Row> positivos = dadosTratados.filter(col("label").equalTo(1.0)).cache();
		Dataset<Row> negativos = dadosTratados.filter(col("label").equalTo(0.0)).cache();
		
		long nPos = positivos.count();
		long nNeg = negativos.count();
		double fracao = (double) nPos / nNeg;
		
		Dataset<Row> amostraNeg = negativos.sample(fracao);
		
		// Modelo para positivos
		Dataset<Row>[] split = positivos.union(amostraNeg).randomSplit(new double[]{0.8, 0.2});
		Dataset<Row> treino = split[0].cache();
		Dataset<Row> teste = split[1].cache();
		
		RandomForestClassifier classificador = new RandomForestClassifier();

		ParamMap[] paramGrid = new ParamGridBuilder()
				.addGrid(classificador.maxDepth(), new int[] {5, 10, 15})
				.addGrid(classificador.maxBins(), new int[] {2, 3, 4})
				.addGrid(classificador.numTrees(), new int[] {10, 25, 50})
				.addGrid(classificador.subsamplingRate(), new double[] {0.2, 0.5, 0.7})
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
		predicoes.groupBy(col("prediction"), col("label")).count().show();

		// Salvando os modelos
		File pasta = new File("data/modelo_positivos");
		if (pasta.exists()) FileUtils.deleteDirectory(pasta);
		modelo.save(pasta.getAbsolutePath());
	
		session.close();
	}

}
