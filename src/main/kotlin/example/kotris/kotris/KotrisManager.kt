package example.kotris.kotris

import java.awt.Color

class KotrisManager {
    var x = 4
    var y = 0

    var currentMino = Mino.I

    var bag = arrayListOf<Mino>()
    var next = arrayListOf<Mino>()
    var hold : Mino? = null
    var holdable = true
    var score = 0

    var board = hashMapOf<Pair<Int, Int>, Mino>()

    fun canMove(newX: Int, newY: Int, newPiece: Mino): Boolean {
        val shape = newPiece.getShape()
        for (i in shape.indices) {
            for (j in shape[i].indices) {
                if (shape[i][j] != 0) {
                    val x = newX + j
                    val y = newY + i
                    if (x !in 0.. 9 || y !in 0.. 19 || board.containsKey(Pair(x,y))) {
                        return false
                    }
                }
            }
        }
        return true
    }

    fun movePiece(newX: Int, newY: Int, newPiece: Mino) {
        if (canMove(newX, newY, newPiece)) {
            x = newX
            y = newY
            currentMino = newPiece
        }
    }

    fun dropPiece() {
        while (canMove(x, y + 1, currentMino)) {
            y++
        }
        lockPiece()
        clearLines()
        resetPiece()
    }

    fun lockPiece() {
        holdable = true
        for (i in currentMino.getShape().indices) {
            for (j in currentMino.getShape()[i].indices) {
                if (currentMino.getShape()[i][j] != 0) {
                    board[Pair(x + j, i + y)] = currentMino
                }
            }
        }
    }

    fun resetPiece() {
        x = 4
        y = 0
        rollPiece()
    }

    fun clearLines() {
        var linesCleared = 0
        for (i in 0..19) {
            if (board.keys.filter { it.second == i }.sumOf { it.first + 1 } == 55) {
                for (k in i downTo 1) {
                    board.keys.filter { it.second == k }.forEach {
                        board.remove(it)
                    }
                    board.keys.filter { it.second == k - 1 }.forEach {
                        board[Pair(it.first, it.second + 1)] = board[it]!!
                    }
                }
                linesCleared++
                lineClear++
            }
        }
        score += linesCleared * 100
    }

    fun clockWise(p : Int, q : Int, mino : Mino) {
        if (mino == Mino.O) return
        if (rotation == 270) rotation = 0
        else rotation += 90
        if (canMove(p, q, mino)) return
        when (rotation) {
            90 -> {
                if (mino == Mino.I) {
                    if (canMove(p - 2, q, mino)) {
                        x -= 2
                        return
                    }
                    if (canMove(p + 1, q, mino)) {
                        x += 1
                        return
                    }
                    if (canMove(p - 2, q + 1, mino)) {
                        x -= 2
                        y += 1
                        return
                    }
                    if (canMove(p + 1, q - 2, mino)) {
                        x += 1
                        y -= 2
                        return
                    }
                    
                } else {
                    if (canMove(p - 1, q, mino)) {
                        x -= 1
                        return
                    }
                    if (canMove(p - 1, q - 1, mino)) {
                        x -= 1
                        y -= 1
                        return
                    }
                    if (canMove(p, q + 2, mino)) {
                        y += 2
                        return
                    }
                    if (canMove(p - 1, q + 2, mino)) {
                        x -= 1
                        y += 2
                        return
                    }
                }
            }
            180 -> {
                if (mino == Mino.I) {
                    if (canMove(p - 1, q, mino)) {
                        x -= 1
                        return
                    }
                    if (canMove(p + 2, q, mino)) {
                        x += 2
                        return
                    }
                    if (canMove(p - 1, q - 2, mino)) {
                        x -= 1
                        y -= 2
                        return
                    }
                    if (canMove(p + 2, q + 1, mino)) {
                        x += 2
                        y -= 1
                        return
                    }
                    
                } else {
                    if (canMove(p + 1, q, mino)) {
                        x += 1
                        return
                    }
                    if (canMove(p + 1, q + 1, mino)) {
                        x += 1
                        y += 1
                        return
                    }
                    if (canMove(p, q - 2, mino)) {
                        y -= 2
                        return
                    }
                    if (canMove(p + 1, q - 2, mino)) {
                        x += 1
                        y -= 2
                        return
                    }
                }
            }
            270 -> {
                if (mino == Mino.I) {
                    if (canMove(p + 2, q, mino)) {
                        x += 2
                        return
                    }
                    if (canMove(p - 1, q, mino)) {
                        x -= 1
                        return
                    }
                    if (canMove(p + 2, q - 1, mino)) {
                        x += 2
                        y -= 1
                        return
                    }
                    if (canMove(p - 1, q + 2, mino)) {
                        x -= 1
                        y += 2
                        return
                    }
                    
                } else {
                    if (canMove(p + 1, q, mino)) {
                        x += 1
                        return
                    }
                    if (canMove(p + 1, q - 1, mino)) {
                        x += 1
                        y -= 1
                        return
                    }
                    if (canMove(p, q + 2, mino)) {
                        y += 2
                        return
                    }
                    if (canMove(p + 1, q + 2, mino)) {
                        x += 1
                        y += 2
                        return
                    }
                }
            }
            0 -> {
                if (mino == Mino.I) {
                    if (canMove(p + 1, q, mino)) {
                        x += 1
                        return
                    }
                    if (canMove(p - 2, q, mino)) {
                        x -= 2
                        return
                    }
                    if (canMove(p + 1, q + 2, mino)) {
                        x += 1
                        y += 2
                        return
                    }
                    if (canMove(p - 2, q - 1, mino)) {
                        x -= 2
                        y -= 1
                        return
                    }

                } else {
                    if (canMove(p - 1, q, mino)) {
                        x -= 1
                        return
                    }
                    if (canMove(p - 1, q + 1, mino)) {
                        x -= 1
                        y += 1
                        return
                    }
                    if (canMove(p, q - 2, mino)) {
                        y -= 2
                        return
                    }
                    if (canMove(p - 1, q - 2, mino)) {
                        x -= 1
                        y -= 2
                        return
                    }
                }
            }
        }
        if (rotation == 0) rotation = 270
        else rotation -= 90
        return
    }

    fun counterClockWise(p : Int, q : Int, mino : Mino) {
        if (mino == Mino.O) return
        if (rotation == 0) rotation = 270
        else rotation -= 90
        if (canMove(p, q, mino)) return
        when (rotation) {
            90 -> {
                if (mino == Mino.I) {
                    if (canMove(p + 1, q, mino)) {
                        x += 1
                        return
                    }
                    if (canMove(p - 2, q, mino)) {
                        x -= 2
                        return
                    }
                    if (canMove(p + 1, q + 2, mino)) {
                        x += 1
                        y += 2
                        return
                    }
                    if (canMove(p - 2, q - 1, mino)) {
                        x -= 2
                        y -= 1
                        return
                    }

                } else {
                    if (canMove(p - 1, q, mino)) {
                        x -= 1
                        return
                    }
                    if (canMove(p - 1, q - 1, mino)) {
                        x -= 1
                        y -= 1
                        return
                    }
                    if (canMove(p, q + 2, mino)) {
                        y += 2
                        return
                    }
                    if (canMove(p - 1, q + 2, mino)) {
                        x -= 1
                        y += 2
                        return
                    }
                }
            }
            180 -> {
                if (mino == Mino.I) {
                    if (canMove(p - 2, q, mino)) {
                        x -= 1
                        return
                    }
                    if (canMove(p + 1, q, mino)) {
                        x += 2
                        return
                    }
                    if (canMove(p - 2, q + 1, mino)) {
                        x -= 2
                        y += 1
                        return
                    }
                    if (canMove(p + 1, q - 2, mino)) {
                        x += 1
                        y -= 2
                        return
                    }

                } else {
                    if (canMove(p - 1, q, mino)) {
                        x -= 1
                        return
                    }
                    if (canMove(p - 1, q + 1, mino)) {
                        x -= 1
                        y += 1
                        return
                    }
                    if (canMove(p, q - 2, mino)) {
                        y -= 2
                        return
                    }
                    if (canMove(p - 1, q - 2, mino)) {
                        x -= 1
                        y -= 2
                        return
                    }
                }
            }
            270 -> {
                if (mino == Mino.I) {
                    if (canMove(p - 1, q, mino)) {
                        x -= 1
                        return
                    }
                    if (canMove(p + 2, q, mino)) {
                        x += 2
                        return
                    }
                    if (canMove(p - 1, q - 2, mino)) {
                        x -= 1
                        y -= 2
                        return
                    }
                    if (canMove(p + 2, q + 1, mino)) {
                        x += 2
                        y += 1
                        return
                    }

                } else {
                    if (canMove(p + 1, q, mino)) {
                        x += 1
                        return
                    }
                    if (canMove(p + 1, q - 1, mino)) {
                        x += 1
                        y -= 1
                        return
                    }
                    if (canMove(p, q + 2, mino)) {
                        y += 2
                        return
                    }
                    if (canMove(p + 1, q + 2, mino)) {
                        x += 1
                        y += 2
                        return
                    }
                }
            }
            0 -> {
                if (mino == Mino.I) {
                    if (canMove(p + 2, q, mino)) {
                        x += 2
                        return
                    }
                    if (canMove(p - 1, q, mino)) {
                        x -= 1
                        return
                    }
                    if (canMove(p + 2, q - 1, mino)) {
                        x += 2
                        y -= 1
                        return
                    }
                    if (canMove(p - 1, q + 2, mino)) {
                        x -= 1
                        y += 2
                        return
                    }
                } else {
                    if (canMove(p + 1, q, mino)) {
                        x += 1
                        return
                    }
                    if (canMove(p + 1, q + 1, mino)) {
                        x += 1
                        y += 1
                        return
                    }
                    if (canMove(p, q - 2, mino)) {
                        y -= 2
                        return
                    }
                    if (canMove(p + 1, q - 2, mino)) {
                        x += 1
                        y -= 2
                        return
                    }
                }
            }
        }
        if (rotation == 270) rotation = 0
        else rotation += 90
        return
    }

    fun halfClockWise(p : Int, q : Int, mino : Mino) {
        if (mino == Mino.O) return
        when (rotation) {
            180 -> rotation = 0
            270 -> rotation = 90
            else -> rotation += 180
        }
        if (canMove(p, q, mino)) return
        when (rotation) {
            90 -> {
                if (canMove(p - 1, q, mino)) {
                    x -= 1
                    return
                }
                if (canMove(p - 1, q - 2, mino)) {
                    x -= 1
                    y -= 2
                    return
                }
                if (canMove(p - 1, q - 1, mino)) {
                    x -= 1
                    y -= 1
                    return
                }
                if (canMove(p, q - 2, mino)) {
                    y -= 2
                    return
                }
                if (canMove(p, q - 1, mino)) {
                    y -= 1
                    return
                }
            }
            180 -> {
                if (canMove(p, q - 1, mino)) {
                    y -= 1
                    return
                }
                if (canMove(p + 1, q - 1, mino)) {
                    x += 1
                    y -= 1
                    return
                }
                if (canMove(p - 1, q + 1, mino)) {
                    x -= 1
                    y += 1
                    return
                }
                if (canMove(p + 1, q, mino)) {
                    x += 1
                    return
                }
                if (canMove(p - 1, q, mino)) {
                    x -= 1
                    return
                }
            }
            270 -> {
                if (canMove(p + 1, q, mino)) {
                    x += 1
                    return
                }
                if (canMove(p + 1, q - 2, mino)) {
                    x += 1
                    y -= 2
                    return
                }
                if (canMove(p + 1, q - 1, mino)) {
                    x += 1
                    y -= 1
                    return
                }
                if (canMove(p, q - 2, mino)) {
                    y -= 2
                    return
                }
                if (canMove(p, q - 1, mino)) {
                    y -= 1
                    return
                }
            }
            0 -> {
                if (canMove(p, q + 1, mino)) {
                    y += 1
                    return
                }
                if (canMove(p - 1, q + 1, mino)) {
                    x -= 1
                    y += 1
                    return
                }
                if (canMove(p + 1, q + 1, mino)) {
                    x += 1
                    y += 1
                    return
                }
                if (canMove(p - 1, q, mino)) {
                    x -= 1
                    return
                }
                if (canMove(p + 1, q, mino)) {
                    x += 1
                }
            }
        }
        if (rotation == 180) rotation = 0
        if (rotation == 270) rotation = 90
        else rotation += 180
        return
    }

    fun rollPiece() {
        rotation = 0
        if (bag.isEmpty()) bag = arrayListOf(
            Mino.Z,
            Mino.I,
            Mino.J,
            Mino.L,
            Mino.O,
            Mino.S,
            Mino.T
        ).shuffled() as ArrayList<Mino>
        while (next.size != 5) {
            next.add(bag[0])
            bag.removeAt(0)
        }
        currentMino = next[0]
        next.removeAt(0)
        next.add(bag[0])
        bag.removeAt(0)
    }

    fun hold() {
        if (!holdable) return
        holdable = false
        if (hold == null) {
            hold = currentMino
            rollPiece()
        } else {
            val temp = currentMino
            currentMino = hold!!
            hold = temp
        }
        x = 4
        y = 0
        rotation = 0
    }

    init {
        rollPiece()
    }
}

var rotation = 0

enum class Mino {
    Z, L, O, S, I, J, T;

    fun getColor() : Color{
        return when(this.name) {
            "Z" -> Color.RED
            "L" -> Color.ORANGE
            "O" -> Color.YELLOW
            "S" -> Color.GREEN
            "I" -> Color.CYAN
            "J" -> Color.BLUE
            "T" -> Color.MAGENTA
            else -> {
                Color.GRAY
            }
        }
    }

    fun getInt() : Int {
        return when(this.name) {
            "Z" -> 1
            "L" -> 2
            "O" -> 3
            "S" -> 4
            "I" -> 5
            "J" -> 6
            "T" -> 7
            else -> {
                8
            }
        }
    }

    fun getRealShape() : Array<Array<Int>> {
        return when (this) {
            I -> arrayOf(
                arrayOf(1, 1, 1, 1)
            )
            O -> arrayOf(
                arrayOf(1, 1),
                arrayOf(1, 1)
            )
            T -> arrayOf(
                arrayOf(0, 1, 0),
                arrayOf(1, 1, 1)
            )
            S -> arrayOf(
                arrayOf(0, 1, 1),
                arrayOf(1, 1, 0)
            )
            Z -> arrayOf(
                arrayOf(1, 1, 0),
                arrayOf(0, 1, 1)
            )
            J -> arrayOf(
                arrayOf(1, 0, 0),
                arrayOf(1, 1, 1)
            )
            L -> arrayOf(
                arrayOf(0, 0, 1),
                arrayOf(1, 1, 1)
            )
        }
    }

    fun getShape() : Array<Array<Int>> {
        if (rotation == 0) return getRealShape()
        if (this == O) return getRealShape()
        if (rotation == 180) return when (this) {
            I -> arrayOf(
                arrayOf(0, 0, 0, 0),
                arrayOf(1, 1, 1, 1)
            )
            T -> arrayOf(
                arrayOf(0, 0, 0),
                arrayOf(1, 1, 1),
                arrayOf(0, 1, 0)
            )
            S -> arrayOf(
                arrayOf(0, 0, 0),
                arrayOf(0, 1, 1),
                arrayOf(1, 1, 0)
            )
            Z -> arrayOf(
                arrayOf(0, 0, 0),
                arrayOf(1, 1, 0),
                arrayOf(0, 1, 1)
            )
            J -> arrayOf(
                arrayOf(0, 0, 0),
                arrayOf(1, 1, 1),
                arrayOf(0, 0, 1)
            )
            L -> arrayOf(
                arrayOf(0, 0, 0),
                arrayOf(1, 1, 1),
                arrayOf(1, 0, 0)
            )

            else -> {throw Error()}
        }
        if (rotation == 90) return when (this) {
            I -> arrayOf(
                arrayOf(0, 0, 1, 0),
                arrayOf(0, 0, 1, 0),
                arrayOf(0, 0, 1, 0),
                arrayOf(0, 0, 1, 0)
            )
            T -> arrayOf(
                arrayOf(0, 1, 0),
                arrayOf(0, 1, 1),
                arrayOf(0, 1, 0)
            )
            S -> arrayOf(
                arrayOf(0, 1, 0),
                arrayOf(0, 1, 1),
                arrayOf(0, 0, 1)
            )
            Z -> arrayOf(
                arrayOf(0, 0, 1),
                arrayOf(0, 1, 1),
                arrayOf(0, 1, 0)
            )
            J -> arrayOf(
                arrayOf(0, 1, 1),
                arrayOf(0, 1, 0),
                arrayOf(0, 1, 0)
            )
            L -> arrayOf(
                arrayOf(0, 1, 0),
                arrayOf(0, 1, 0),
                arrayOf(0, 1, 1)
            )

            else -> {throw Error()}
        }
        if (rotation == 270) return when (this) {
            I -> arrayOf(
                arrayOf(0, 1, 0, 0),
                arrayOf(0, 1, 0, 0),
                arrayOf(0, 1, 0, 0),
                arrayOf(0, 1, 0, 0)
            )
            T -> arrayOf(
                arrayOf(0, 1, 0),
                arrayOf(1, 1, 0),
                arrayOf(0, 1, 0)
            )
            S -> arrayOf(
                arrayOf(1, 0, 0),
                arrayOf(1, 1, 0),
                arrayOf(0, 1, 0)
            )
            Z -> arrayOf(
                arrayOf(0, 1, 0),
                arrayOf(1, 1, 0),
                arrayOf(1, 0, 0)
            )
            J -> arrayOf(
                arrayOf(0, 1, 0),
                arrayOf(0, 1, 0),
                arrayOf(1, 1, 0)
            )
            L -> arrayOf(
                arrayOf(1, 1, 0),
                arrayOf(0, 1, 0),
                arrayOf(0, 1, 0)
            )

            else -> {throw Error()}
        }
        println(rotation)
        throw Exception()
    }
}