package example.geneticAlgorithm.kotris.kotris

import example.geneticAlgorithm.kotris.a_library.swp.SwpCanvas
import example.geneticAlgorithm.kotris.a_library.swp.SwpWindow
import kotlinAILibrary.GeneticAlgorithm
import kotlinAILibrary.NeuralNetwork
import java.awt.Color

var lineClear = 0
var lineSuccess = hashSetOf<Int>()

fun main() {
    val window = SwpWindow("Kotris", 1920, 1080)
    val canvas = SwpCanvas(Color.BLACK)

    canvas.paint {  }
    val mutationRate = 2
    val geneticAlgorithm = GeneticAlgorithm(List(3000) { NeuralNetwork(210, listOf(109), 8) }, {a, _ ->
        lineClear = 0
        lineSuccess = hashSetOf()
        val start = System.currentTimeMillis()
        val manager = KotrisManager()
        a.memoList.add(0.0)
        a.memoList.add(0)
        canvas.paint {
            val tick = (System.currentTimeMillis() - start) / 50
            KotrisRenderer(it, manager, tick).render()
        }

        for (i in 0 until 100) {
            val doubleArray = DoubleArray(210) {0.0}
            var breakit = false
            lineSuccess = hashSetOf()
            manager.board.forEach { (t, u) ->
                if (t.second <= 5 && u.getInt() != 8) breakit = true
                if (t.second >= 18 && lineClear < 4 && u.getInt() != 8) lineSuccess.add(t.first + (t.second * 10))
                doubleArray[t.second * 10 + t.first] = u.getInt().toDouble()
            }

            if (breakit) break

            doubleArray[200] = manager.currentMino.getInt().toDouble()
            doubleArray[201] = manager.x.toDouble()
            doubleArray[202] = manager.y.toDouble()
            if (manager.hold == null) doubleArray[203] = 8.0 else doubleArray[203] = manager.hold!!.getInt().toDouble()
            doubleArray[204] = manager.next[0].getInt().toDouble()
            doubleArray[205] = manager.next[1].getInt().toDouble()
            doubleArray[206] = manager.next[2].getInt().toDouble()
            doubleArray[207] = manager.next[3].getInt().toDouble()
            doubleArray[208] = manager.next[4].getInt().toDouble()
            doubleArray[209] = 10 * i.toDouble()

            val selected = a.forward(doubleArray).withIndex().maxByOrNull { it.value }!!.index
            when (selected) {
                0 -> manager.movePiece(manager.x - 1, manager.y, manager.currentMino)
                1 -> manager.movePiece(manager.x + 1, manager.y, manager.currentMino)
                2 -> manager.movePiece(manager.x, manager.y + 1, manager.currentMino)
                3 -> manager.clockWise(manager.x, manager.y, manager.currentMino)
                4 -> manager.counterClockWise(manager.x, manager.y, manager.currentMino)
                5 -> manager.dropPiece()
                6 -> manager.hold()
                7 -> manager.halfClockWise(manager.x, manager.y, manager.currentMino)
            }

            a.memoList[1] = lineSuccess.size + (lineClear * 100)

        }

        if (lineClear >= 4) {
            a.memoList[0] = (System.currentTimeMillis() - start) / 1000.0
            println("Line Cleared")
            (System.currentTimeMillis() - start) / 1000.0
        } else {
            a.memoList[0] = Int.MAX_VALUE.toDouble() - (a.memoList[1].toString().toInt() * 100000)
            println(a.memoList[1].toString())
            Int.MAX_VALUE.toDouble() - (a.memoList[1].toString().toInt() * 100000)
        }
    }, {a, b -> a.sortedBy { b[a.indexOf(it)] }.take(a.size / 2)}, null, mutationRate.toDouble())

    window.setCanvas(canvas)
    window.timerRun()
    window.enableWindow()
    val bestAIs = arrayListOf<NeuralNetwork>()
    geneticAlgorithm.startEvolve()
    repeat(200000) { generation ->
        val a = geneticAlgorithm.p.map { it.memoList[0] as Double }.toDoubleArray()
        bestAIs.add(geneticAlgorithm.p[a.withIndex().minBy { it.value }.index])
        val bestAI = bestAIs.last()

        println("Generation $generation complete")
        println("Score : ${bestAI.memoList[0].toString().toDouble()}")
        println("GoodPoint : ${bestAI.memoList[1].toString().toInt()}")
        geneticAlgorithm.evolve()
    }
    val bestAI = bestAIs.minByOrNull { it.memoList[0].toString().toDouble() }!!

    println("Score : ${bestAI.memoList[0].toString().toDouble()}")
}