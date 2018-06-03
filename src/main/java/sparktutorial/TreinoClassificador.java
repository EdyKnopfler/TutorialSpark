package sparktutorial;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
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
		
		// Treinamento do classificador
		Dataset<Row>[] splits = dataset.randomSplit(new double[]{0.8, 0.2});
		Dataset<Row> treino = splits[0].cache();
		Dataset<Row> teste = splits[1].cache();
		
		RandomForestClassifier classificador = new RandomForestClassifier()
				.setLabelCol("label").setFeaturesCol("features");

		ParamMap[] paramGrid = new ParamGridBuilder()
				.addGrid(classificador.maxDepth(), new int[] {10, 15, 20, 25, 30})
				.addGrid(classificador.maxBins(), new int[] {16, 32, 46})
				.build();

		MulticlassClassificationEvaluator avaliador = new MulticlassClassificationEvaluator();
		
		CrossValidator cv = new CrossValidator()
				.setEstimator(classificador)
				.setEvaluator(avaliador)
				.setEstimatorParamMaps(paramGrid)
				.setNumFolds(10);
		
		CrossValidatorModel modelo = cv.fit(treino);
		
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
