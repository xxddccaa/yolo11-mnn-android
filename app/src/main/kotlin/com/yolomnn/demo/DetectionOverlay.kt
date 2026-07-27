package com.yolomnn.demo

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.yolomnn.Detection

/**
 * Draws detection boxes + labels over content sized [bitmapWidth]x[bitmapHeight].
 *
 * [fitCenter] = true maps boxes as an ImageView with scaleType=fitCenter (used
 * for the album/still-image mode). [fitCenter] = false uses centerCrop / fill
 * mapping (used for the CameraX PreviewView which fills the view).
 */
class DetectionOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var detections: List<Detection> = emptyList()
    private var bitmapWidth = 0
    private var bitmapHeight = 0
    private var fitCenter = true

    private val palette = intArrayOf(
        Color.rgb(0, 200, 83), Color.rgb(33, 150, 243), Color.rgb(255, 152, 0),
        Color.rgb(233, 30, 99), Color.rgb(156, 39, 176), Color.rgb(0, 188, 212),
        Color.rgb(255, 87, 34), Color.rgb(76, 175, 80),
    )

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 34f
        isFakeBoldText = true
    }
    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    fun setResults(detections: List<Detection>, bitmapWidth: Int, bitmapHeight: Int, fitCenter: Boolean) {
        this.detections = detections
        this.bitmapWidth = bitmapWidth
        this.bitmapHeight = bitmapHeight
        this.fitCenter = fitCenter
        invalidate()
    }

    fun clear() {
        detections = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty() || bitmapWidth <= 0 || bitmapHeight <= 0) return

        val scale: Float
        val dx: Float
        val dy: Float
        if (fitCenter) {
            scale = minOf(width.toFloat() / bitmapWidth, height.toFloat() / bitmapHeight)
            dx = (width - bitmapWidth * scale) / 2f
            dy = (height - bitmapHeight * scale) / 2f
        } else {
            // centerCrop: scale to fill, center-align, allow overflow
            scale = maxOf(width.toFloat() / bitmapWidth, height.toFloat() / bitmapHeight)
            dx = (width - bitmapWidth * scale) / 2f
            dy = (height - bitmapHeight * scale) / 2f
        }

        val rectF = RectF()
        for (d in detections) {
            val color = palette[d.classId % palette.size]
            boxPaint.color = color
            labelBgPaint.color = color
            rectF.set(
                d.box.left * scale + dx,
                d.box.top * scale + dy,
                d.box.right * scale + dx,
                d.box.bottom * scale + dy,
            )
            canvas.drawRect(rectF, boxPaint)

            val text = "${d.label} ${(d.score * 100).toInt()}%"
            val tw = textPaint.measureText(text)
            val th = textPaint.fontMetrics.let { it.descent - it.ascent }
            var labelTop = rectF.top - th
            if (labelTop < 0) labelTop = rectF.top
            canvas.drawRect(rectF.left, labelTop, rectF.left + tw + 12f, labelTop + th, labelBgPaint)
            canvas.drawText(text, rectF.left + 6f, labelTop - textPaint.fontMetrics.ascent, textPaint)
        }
    }
}
