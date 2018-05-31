package sparktutorial;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.SparkSession.Builder;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.streaming.StreamingQueryException;

import static org.apache.spark.sql.functions.when;
import static org.apache.spark.sql.functions.col;

public class ClassificacaoStreaming {

	public static void main(String[] args) {
		// Inicialização
		Builder builder = SparkSession.builder().appName("MonitorDeSpam");
		if (args.length > 0 && args[0].equals("-l")) builder.master("local[*]");
		SparkSession session = builder.getOrCreate();
		Logger.getRootLogger().setLevel(Level.WARN);
		
		// Previamente treinados
		CrossValidatorModel modelo = CrossValidatorModel.load("data/modelo_atual");
		
		// Inicialização do streaming
		// Antes execute:
		// $ nc -lk -p 9999
		Dataset<Row> mensagens = session
				  .readStream()
				  .format("socket")
				  .option("host", "localhost")
				  .option("port", 9999)
				  .load();
		
		// Classificação
		Dataset<Row> classificacoes = 
				modelo
					.transform(mensagens)
					.select(
						col("value"), 
						when(col("prediction").equalTo(0.0), "HAM").otherwise("SPAM").as("predicao")
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
