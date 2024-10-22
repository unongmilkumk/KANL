package example.kotris.kotris

import java.awt.Color
import java.awt.Graphics

class KotrisRenderer(val graphics: Graphics, val manager: KotrisManager, val tick : Long) {
    fun render() {
        graphics.color = Color(255, 255, 255, 100)
        for (i in (0..10)) graphics.drawLine(i * 30 + 810, 100, i * 30 + 810, 700)
        for (i in (0..20)) graphics.drawLine(810, i * 30 + 100, 1110, i * 30 + 100)

        var temp = 0
        while (manager.canMove(manager.x, manager.y + temp, manager.currentMino)) {
            temp++
        }
        graphics.color = Color(128, 128, 128, 128)
        val qwer = manager.currentMino.getShape()
        qwer.indices.forEach { j ->
            qwer[j].indices.forEach { i ->
                if (qwer[j][i] != 0)
                    graphics.fillRect((manager.x + i) * 30 + 810, (manager.y + j + temp - 1) * 30 + 100, 30, 30)
            }
        }

        graphics.color = manager.currentMino.getColor()
        qwer.indices.forEach { j ->
            qwer[j].indices.forEach { i ->
                if (qwer[j][i] != 0)
                    graphics.fillRect((manager.x + i) * 30 + 810, (manager.y + j) * 30 + 100, 30, 30)
            }
        }

        for (i in (0.. 10)) {
            for (j in (0 ..20)) {
                graphics.color = (manager.board[Pair(i, j)] ?: continue).getColor()
                graphics.fillRect(i * 30 + 810, j * 30 + 100, 30, 30)
            }
        }

        if (manager.hold != null) {
            val holdPiece = manager.hold!!
            graphics.color = if (manager.holdable) holdPiece.getColor() else Color.GRAY
            for (i in holdPiece.getRealShape().indices) {
                for (j in holdPiece.getRealShape()[i].indices) {
                    if (holdPiece.getRealShape()[i][j] != 0) {
                        graphics.fillRect(j * 30 + 760 - holdPiece.getRealShape()[i].size * 30, i * 30 + 100, 30, 30)
                    }
                }
            }
        }

        val nextPieces = manager.next.toList()
        for (n in nextPieces.indices) {
            val piece = nextPieces[n]
            graphics.color = piece.getColor()
            for (i in piece.getRealShape().indices) {
                for (j in piece.getRealShape()[i].indices) {
                    if (piece.getRealShape()[i][j] != 0) {
                        graphics.fillRect(j * 30 + 1160, (n * 5 + i) * 30 + 100, 30, 30)
                    }
                }
            }
        }
    }
}