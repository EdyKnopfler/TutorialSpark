package sparktutorial;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

// https://github.com/felipecruz/exemplos/tree/master/data
public class Rdd {
	
	public static void main(String[] args) {
		// Inicialização
		SparkConf config = new SparkConf();
		config.setAppName("QuemEstaRoubando");
		if (args.length > 3 && args[3].equals("-l")) config.setMaster("local[*]");
		JavaSparkContext context = new JavaSparkContext(config);
		
		// Instituições
		JavaRDD<String> instituicoesRaw = context.textFile(args[0]);
		JavaPairRDD<Long, String> instituicoes = instituicoesRaw.mapToPair(linha -> {
			String[] campos = linha.split(";");
			long idInstituicao = Long.valueOf(campos[0]) ;
			String nome = campos[2];
			return new Tuple2<>(idInstituicao, nome);
		});

		// Execuções financeiras
		JavaRDD<String> execucoesRaw = context.textFile(args[1]);
		JavaPairRDD<Long, Double> execucoes = execucoesRaw.mapToPair(linha -> {
			String[] campos = linha.split(";");
			long idInstituicao = Long.valueOf(campos[2]);
			double contratado = Double.valueOf(campos[5]);
			double total = Double.valueOf(campos[6]);
			return new Tuple2<>(idInstituicao, total - contratado);
		}).filter(tupla -> tupla._1 != -1);
		
		// Quem está ultrapassando o valor?
		JavaPairRDD<Long, Double> totais = execucoes.reduceByKey((a, b) -> a + b);
		JavaPairRDD<Long, Double> ladroes = totais.filter(tupla -> tupla._2 > 0.00);
		
		// Liga os RDDs
		JavaPairRDD<Long, Tuple2<Double, String>> juncao = ladroes.join(instituicoes);
		
		// Salva
		JavaRDD<String> csv = 
				juncao.map(tupla -> tupla._1 + ";" + tupla._2._1 + ";" + tupla._2._2).cache();
		DateFormat instante = new SimpleDateFormat("yyyy_MM_dd_hh_mm_ss_SSS");
		csv.saveAsTextFile(args[2] + "_" + instante.format(new Date()));
		
		// Coleta alguns e exibe
		List<String> amostras = csv.take(10);
		for (String a : amostras)
			System.out.println(a);
		
		context.close();
	}
	
}
