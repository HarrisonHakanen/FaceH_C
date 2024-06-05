#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <math.h>
#include <filesystem>

#include "global_kmeans.h"
#include "functions_kmeans.h"


void escreverArquivoKmeans(std::string nomeArquivo, auto conteudo) {

	if (std::filesystem::exists(nomeArquivo)) {

		std::ofstream arquivoCadastro(nomeArquivo, std::ios::app);

		if (arquivoCadastro.is_open()) {
			arquivoCadastro << conteudo;
			arquivoCadastro.close();
		}

	}
	else {
		std::ofstream arquivoCadastro(nomeArquivo);
		arquivoCadastro << conteudo;
		arquivoCadastro.close();
	}

}


Centroid calculoDistancia(std::vector<float> dimensoes, Centroid centroid) {

	float distancia = 0;
	float potencia = 0;

	for (int i = 0; i < quantDimensoes; i++) {

		potencia += pow(dimensoes[i] - centroid.dimensoes[i], 2);
	}
	centroid.distancia = sqrt(potencia);

	return centroid;
}


std::vector<Registro> classificaRegistros(std::vector<Registro> registros, std::vector<Centroid> centroides) {

		
	for (int indexReg = 0; indexReg < registros.size(); indexReg++) {


		if (registros[indexReg].classe.id == 0) {

			for (int indexCent = 0; indexCent < centroides.size(); indexCent++) {

				centroides[indexCent] = calculoDistancia(registros[indexReg].dimensoes, centroides[indexCent]);				
			}

			int indexMenorCentroid = -1;
			float menorDistancia = -1;
			for (int indexCent = 0; indexCent < centroides.size(); indexCent++) {

				if (indexCent == 0) {

					indexMenorCentroid = indexCent;
					menorDistancia = centroides[indexCent].distancia;
				}
				else {
					if (centroides[indexCent].distancia < menorDistancia) {

						indexMenorCentroid = indexCent;
						menorDistancia = centroides[indexCent].distancia;
					}
				}
			}

			registros[indexReg].classe = centroides[indexMenorCentroid];
		}
	}	
		
	return registros;
}


Centroid classificaRegistro(std::vector<float> registros, std::vector<Centroid> centroides) {

	for (int indexCent = 0; indexCent < centroides.size(); indexCent++) {

		centroides[indexCent] = calculoDistancia(registros, centroides[indexCent]);
	}

	int indexMenorCentroid = -1;
	float menorDistancia = -1;
	for (int indexCent = 0; indexCent < centroides.size(); indexCent++) {

		if (indexCent == 0) {

			indexMenorCentroid = indexCent;
			menorDistancia = centroides[indexCent].distancia;
		}
		else {
			if (centroides[indexCent].distancia < menorDistancia) {
				indexMenorCentroid = indexCent;
				menorDistancia = centroides[indexCent].distancia;
			}
		}
	}

	return centroides[indexMenorCentroid];				
}




std::vector<Registro> separaOsClusters(std::vector<Registro> registros, int indexReg) {

	std::vector<Registro> registrosCluster;
	registrosCluster.push_back(registros[indexReg]);

	if (indexReg + 1 < registros.size()) {

		for (int indexReg2 = indexReg + 1; indexReg2 < registros.size(); indexReg2++) {

			if (registros[indexReg].classe.id == registros[indexReg2].classe.id) {

				registrosCluster.push_back(registros[indexReg2]);
			}
		}
	}

	return registrosCluster;
}


RetornoCentroides reposicionaClusters(std::vector<Registro> registrosCluster, std::vector<Centroid>centroides) {


	bool alterou = false;
	std::vector<float>dimensoesVar;


	for (int i = 0; i < quantDimensoes; i++) {
		dimensoesVar.push_back(0);
	}

	//Somatória de todas as dimensoes de cada registro.
	for (int indexClus = 0; indexClus < registrosCluster.size(); indexClus++) {

		for (int indexDimen = 0; indexDimen < registrosCluster[indexClus].dimensoes.size(); indexDimen++) {

			float valor = registrosCluster[indexClus].dimensoes[indexDimen];
			dimensoesVar[indexDimen] += valor;
		}
	}

	//Média das somatórias de cada registro.
	for (int indexDimen = 0; indexDimen < quantDimensoes; indexDimen++) {

		dimensoesVar[indexDimen] = dimensoesVar[indexDimen] / registrosCluster.size();
	}

	//Reposiciona os cluster com base na media.
	for (int indexCent = 0; indexCent < centroides.size(); indexCent++) {

		//A indexação sempre é 0 pois esse vetor sempre vai pertencer a um cluster
		if (registrosCluster[0].classe.id == centroides[indexCent].id) {


			for (int indexDimen = 0; indexDimen < quantDimensoes; indexDimen++) {

				if (centroides[indexCent].dimensoes[indexDimen] != dimensoesVar[indexDimen]) {

					alterou = true;
				}
				centroides[indexCent].dimensoes[indexDimen] = dimensoesVar[indexDimen];
			}
		}
	}

	RetornoCentroides retorno = { centroides,alterou };

	return retorno;
}


void criarCentroides(std::string arquivoCentroides) {


	if (!std::filesystem::exists(arquivoCentroides)) {

		std::vector<Centroid> centroides;

		for (int i = 0; i < qtdCentroides; i++) {

			Centroid centroid = { i,geraDimensoesComIntervalo(100,800),-1 };
			centroides.push_back(centroid);
		}


		for (int indexCent = 0; indexCent < qtdCentroides; indexCent++) {

			std::string conteudo = "Centroide" + std::string("[") + std::to_string(indexCent) + "[";


			for (int indexDimen = 0; indexDimen < centroides[indexCent].dimensoes.size(); indexDimen++) {

				if (indexDimen < centroides[indexCent].dimensoes.size() - 1) {

					conteudo += std::to_string(centroides[indexCent].dimensoes[indexDimen]) + "{";
				}
				else {

					conteudo += std::to_string(centroides[indexCent].dimensoes[indexDimen]);
				}
			}

			escreverArquivoKmeans(arquivoCentroides, conteudo);
		}
	}
}

std::vector<float>geraDimensoes() {

	std::vector<float> dimensoes;
	for (int i = 0; i < quantDimensoes; i++) {
		dimensoes.push_back((float)std::rand() / (float)(RAND_MAX / valorMaximo));
	}

	return dimensoes;
}


std::vector<float>geraDimensoesComIntervalo(int rangeMax, int rangeMin) {

	std::vector<float> dimensoes;
	int min = 1;
	int max = 4;

	int minQtdZ = 1;
	int maxQtdZ = 2;


	auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	srand(static_cast<unsigned>(seed));


	for (int x = 0; x < quantDimensoes; x++) {
		double randNum = rand() % (rangeMax - rangeMin + 1) + rangeMin;
		int randImpPar = rand() % (max - min + 1) + min;


		if (randImpPar % 2 == 0) {			
			dimensoes.push_back(std::stod("0.0" + std::to_string(randNum)));
		}
		else {
			dimensoes.push_back(std::stod("-0.0" + std::to_string(randNum)));		
		}
	}

	return dimensoes;
}