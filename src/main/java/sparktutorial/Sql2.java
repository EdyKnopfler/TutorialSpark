package sparktutorial;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.SparkSession.Builder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.col;

public class Sql2 {
	
	public static void main(String[] args) {
		// Inicialização
		Builder builder = SparkSession.builder().appName("QuemEstaRoubandoSQL");
		if (args.length > 3 && args[3].equals("-l")) builder.master("local[*]");
		SparkSession session = builder.getOrCreate();
		
		List<StructField> campos;
		StructType esquema;
		
		// Instituições
		campos = new ArrayList<>();
		campos.add(DataTypes.createStructField("id_instituicao", DataTypes.LongType, true));
		campos.add(DataTypes.createStructField("id_tipo", DataTypes.LongType, true));
		campos.add(DataTypes.createStructField("nome", DataTypes.StringType, true));
		campos.add(DataTypes.createStructField("cnpj", DataTypes.StringType, true));
		
		esquema = DataTypes.createStructType(campos);

		Dataset<Row> instituicoes = 
				session
					.read()
					.format("csv")
					.option("delimiter", ";")
					.option("header", "false")
					.schema(esquema)
					//.option("inferSchema", "true")
					.load(args[0]);

		// instituicoes.show();
		
		// Execuções financeiras
		campos = new ArrayList<>();
		campos.add(DataTypes.createStructField("id_execucao", DataTypes.LongType, true));
		campos.add(DataTypes.createStructField("id_empreendimento", DataTypes.LongType, true));
		campos.add(DataTypes.createStructField("id_instituicao", DataTypes.LongType, true));
		campos.add(DataTypes.createStructField("id_pessoa_fisica", DataTypes.LongType, true));
		campos.add(DataTypes.createStructField("id_licitacao", DataTypes.LongType, true));
		campos.add(DataTypes.createStructField("valor_contrato", DataTypes.createDecimalType(), true));
		campos.add(DataTypes.createStructField("valor_total", DataTypes.createDecimalType(), true));
		campos.add(DataTypes.createStructField("data_assinatura", DataTypes.DateType, true));
		campos.add(DataTypes.createStructField("data_inicio", DataTypes.DateType, true));
		campos.add(DataTypes.createStructField("data_final", DataTypes.DateType, true));
		
		esquema = DataTypes.createStructType(campos);
		
		Dataset<Row> execucoes = 
				session
					.read()
					.format("csv")
					.option("delimiter", ";")
					.option("header", "false")
					.option("dateFormat", "dd/MM/yyyy")
					.schema(esquema)
					//.option("inferSchema", "true")
					.load(args[1]);
					
		
		// execucoes.show();
		
		// SQL API
		Dataset<Row> ladroes = 
			execucoes
				.select(col("id_instituicao"), 
						col("valor_total").minus(col("valor_contrato")).as("roubo"))
				.filter(col("id_instituicao").notEqual(-1))
				.join(instituicoes, "id_instituicao")
				.select(col("nome"), col("cnpj"), col("roubo"))
				.groupBy(col("nome"), col("cnpj"))
				.sum("roubo")
				.filter(col("sum(roubo)").gt(0.00));
		
		
		
		/*
		instituicoes.createOrReplaceTempView("instituicoes");
		execucoes.createOrReplaceTempView("execucoes");
		
		Dataset<Row> ladroes = session.sql(
			"SELECT i.nome, i.cnpj, SUM(e.valor_total - e.valor_contrato) AS roubo " +
			"FROM instituicoes i NATURAL JOIN execucoes e " +
			"WHERE e.id_instituicao <> -1 " +
			"GROUP BY i.nome, i.cnpj " +
			"HAVING SUM(e.valor_total - e.valor_contrato) > 0.00 "
		).cache();
		*/

		// Saídas
		//ladroes.show();
		DateFormat instante = new SimpleDateFormat("yyyy_MM_dd_hh_mm_ss_SSS");
		ladroes.write().csv(args[2] + instante.format(new Date()));
		
		session.close();
	}

}
