#include "tablewidgetheaderpainting.h"
#include "qpainter.h"

void tablewidgetheaderpainting::paintSection(QPainter * painter, const QRect & rect, int logicalIndex) const
{
    QVariant bg = model()->headerData(logicalIndex, Qt::Horizontal, Qt::BackgroundRole);
    painter->save();
    QHeaderView::paintSection(painter, rect, logicalIndex);
    painter->restore();
    if(bg.isValid())
        painter->fillRect(rect, bg.value<QBrush>());
}
