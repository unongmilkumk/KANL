package kotlinAILibrary

class GeneticAlgorithm(population : List<NeuralNetwork>,
                       val score : (NeuralNetwork, List<NeuralNetwork>) -> (Double),
                       val select : (List<NeuralNetwork>, List<Double>) -> (List<NeuralNetwork>),
                       val crossover : (Pair<NeuralNetwork, NeuralNetwork>) -> NeuralNetwork,
                       val mutate : (NeuralNetwork) -> NeuralNetwork
) {
    var p = population
    var pSize = population.size

    fun evolve() : Pair<DoubleArray, NeuralNetwork> {
        val fitnessScores = DoubleArray(pSize) { 0.0 }
        var best = 0
        p.withIndex().forEach { (index, nn) ->
            fitnessScores[index] = score(nn, p)
        }

        val selected = select(p, fitnessScores.toList())

        val newPopulation = mutableListOf<NeuralNetwork>()
        for (i in 0 until pSize) {
            val parent1 = selected.random()
            val parent2 = selected.random()
            newPopulation.add(crossover(Pair(parent1, parent2)))
        }
        p = newPopulation.map { mutate(it) }

        p.withIndex().forEach { (index, nn) ->
            fitnessScores[index] = score(nn, p)
            if (score(nn, p) > score(p[best], p)) best = index
        }

        return Pair(fitnessScores, p[best])
    }
}