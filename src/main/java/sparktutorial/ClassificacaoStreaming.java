package sparktutorial;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.SparkSession.Builder;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.streaming.StreamingQueryException;

import static org.apache.spark.sql.functions.when;
import static org.apache.spark.sql.functions.col;

public class ClassificacaoStreaming {

	public static void main(String[] args) {
		// Inicialização
		Builder builder = SparkSession.builder().appName("ClassificadorTempoQuaseReal");
		if (args.length > 0 && args[0].equals("-l")) builder.master("local[*]");
		SparkSession session = builder.getOrCreate();
		Logger.getRootLogger().setLevel(Level.WARN);
		
		// Previamente treinado
		CrossValidatorModel modelo = CrossValidatorModel.load("data/modelo_atual");
		
		// Inicialização do streaming
		// Antes execute:
		// $ nc -lk -p 9999
		Dataset<Row> novosClientesCSV = session
				  .readStream()
				  .format("socket")
				  .option("host", "localhost")
				  .option("port", 9999)
				  .load()
				  
				  // TODO aqui é lido como string crua, tem que criar uma extração à parte :P
				  // No lugar do rótulo, colocar uma coluna "name" para a identificação do cara na saída
				  
				  .map(new FeatureExtractor(), RowEncoder.apply(FeatureExtractor.SCHEMA_TREINO));
		
		// Classificação
		Dataset<Row> classificacoes = modelo
					.transform(novosClientesCSV)
					.select(
						col("name"), 
						when(col("prediction").equalTo(0.0), "NO").otherwise("YES").as("predicao")
					);
		
		// Visualizando
		try {
			StreamingQuery query = classificacoes.writeStream()
					.format("console")
					.start();
			query.awaitTermination();
		} 
		catch (StreamingQueryException e) {
			System.out.println("Bye.");
		}
		
		session.close();
	}

}
