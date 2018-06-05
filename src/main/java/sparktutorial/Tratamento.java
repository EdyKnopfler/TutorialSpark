package sparktutorial;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.SparkSession.Builder;

import static org.apache.spark.sql.functions.col;

public class Tratamento {

	public static void main(String[] args) throws IOException {
		// Inicialização e carregamento
		Builder builder = SparkSession.builder().appName("ClassificadorBancario");
		if (args.length > 1 && args[1].equals("-l")) builder.master("local[*]");
		SparkSession session = builder.getOrCreate();
		Logger.getRootLogger().setLevel(Level.WARN);
		
		Dataset<Row> dados = session
				.read()
				.format("csv")
				.option("delimiter", ";")
				.option("header", "true")
				.schema(Dados.SCHEMA_DADOS)
				.load(args[0])
				.drop("duration");

		// Pipeline
		List<PipelineStage> estagios = new ArrayList<>();
		
		// 1. Indexar variáveis categóricas
		String[] categoricas = new String[] {
				"job", "marital", "education", "default", "housing", "loan", "contact", "month",
				"day_of_week", "poutcome", "y"
		};

		for (int i = 0; i < categoricas.length; i++) {
			String nomeAnt = categoricas[i];
			categoricas[i] = categoricas[i] + "Index";
			StringIndexer indexador = new StringIndexer()
					.setInputCol(nomeAnt)
					.setOutputCol(categoricas[i]);
			estagios.add(indexador);
		}
		
		// 2. Criar dummies
		String[] dummies = new String[categoricas.length - 1];  // Excluir o rótulo ("y")
		for (int j = 0; j < categoricas.length - 1; j++)
			dummies[j] = categoricas[j] + "Dummy";
		
		OneHotEncoderEstimator theDummynator = new OneHotEncoderEstimator()
				  .setInputCols(Arrays.copyOfRange(categoricas, 0, categoricas.length - 1))
				  .setOutputCols(dummies);
		estagios.add(theDummynator);
		
		// 3. Juntar variáveis em um único Vector
		String[] numericas = new String[] {
				"age", "campaign", "pdays", "previous", "emp_var_rate", "cons_price_idx",
				"cons_conf_idx", "euribor3m", "nr_employed"
		};
		String[] tudo = new String[numericas.length + dummies.length];
		
		int i = 0;
		for (int j = 0; j < numericas.length; j++, i++)
			tudo[i] = numericas[j];
		for (int j = 0; j < dummies.length; j++, i++)
			tudo[i] = dummies[j];
		
		System.out.println(Arrays.asList(tudo));
		
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(tudo).setOutputCol("features");
		estagios.add(assembler);
		
		// E lá vamos nós :)
		Pipeline pipeline = new Pipeline().setStages(estagios.toArray(new PipelineStage[] {}));
		PipelineModel tratamento = pipeline.fit(dados);
		
		File pasta = new File("data/tratamento");
		if (pasta.exists()) FileUtils.deleteDirectory(pasta);
		tratamento.save(pasta.getAbsolutePath());
		
		Dataset<Row> dadosTratados = tratamento.transform(dados)
				.select(col("features"), col("yIndex").as("label")).cache();
		dadosTratados.show();
		
		pasta = new File("data/tratados");
		if (pasta.exists()) FileUtils.deleteDirectory(pasta);
		dadosTratados.write().parquet("data/tratados");
		
		session.close();
	}

}
