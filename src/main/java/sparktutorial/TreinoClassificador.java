package sparktutorial;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.SparkSession.Builder;

import scala.collection.Iterable;
import scala.collection.JavaConverters;
import scala.collection.convert.Decorators.AsScala;

import static org.apache.spark.sql.functions.col;

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

		// Amostragem: existem MUITO mais negativos que positivos!
		Dataset<Row> positivos = dadosTratados.filter(col("label").equalTo(1.0)).cache();
		Dataset<Row> negativos = dadosTratados.filter(col("label").equalTo(0.0)).cache();
		
		long nPos = positivos.count();
		long nNeg = negativos.count();
		long razao = nNeg / nPos;
		
		Dataset<Row> amostra = negativos;
		for (int i = 0; i < razao; i++)
			amostra = amostra.union(positivos);
		
		// Modelo para positivos
		Dataset<Row>[] split = amostra.randomSplit(new double[]{0.8, 0.2});
		Dataset<Row> treino = split[0].cache();
		Dataset<Row> teste = split[1]/*.union(sobra)*/.cache();
		
		RandomForestClassifier classificador = new RandomForestClassifier();
		
		List<double[]> thresholds = Arrays.asList(new double[][] {
			new double[] { .1, .9 },
			new double[] { .2, .8 },	
			new double[] { .4, .6 },	
			new double[] { .6, .4 },	
			new double[] { .8, .2 },
			new double[] { .9, .1 },
		});

		AsScala<Iterable<double[]>> converter = 
				JavaConverters.iterableAsScalaIterableConverter(thresholds);
		Iterable<double[]> doJeitoQueODiaboGosta = converter.asScala();
		ParamMap[] paramGrid = new ParamGridBuilder()
				.addGrid(classificador.thresholds(), doJeitoQueODiaboGosta)
				.addGrid(classificador.maxDepth(), new int[] {5, 10, 15})
				.addGrid(classificador.maxBins(), new int[] {10, 20, 50, 75})
				.addGrid(classificador.numTrees(), new int[] {15, 20, 50})
				.build();

		TrainValidationSplit validador = new TrainValidationSplit()
				.setEstimator(classificador)
				.setEvaluator(new MulticlassClassificationEvaluator())
				.setEstimatorParamMaps(paramGrid)
				.setTrainRatio(0.8);

		TrainValidationSplitModel modelo = validador.fit(treino);
		
		// Teste
		Dataset<Row> predicoes = modelo.transform(teste);
		double metrica = validador.getEvaluator().evaluate(predicoes);
		System.out.println(
				((MulticlassClassificationEvaluator) validador.getEvaluator()).getMetricName() +
				": " + metrica);
		RandomForestClassificationModel escolhido = (RandomForestClassificationModel) modelo.bestModel();
		System.out.println("Thresholds: " + Arrays.asList(escolhido.getThresholds()));
		System.out.println("MaxDepth: " + escolhido.getMaxDepth());
		System.out.println("MaxBins: " + escolhido.getMaxBins());
		System.out.println("NumTrees: " + escolhido.getNumTrees());
		predicoes.groupBy(col("prediction"), col("label")).count().show();

		// Salvando os modelos
		File pasta = new File("data/modelo");
		if (pasta.exists()) FileUtils.deleteDirectory(pasta);
		modelo.save(pasta.getAbsolutePath());
	
		session.close();
	}

}
