#include "selectioncontroldelegate.h"
#include "qheaderview.h"

SelectionControlDelegate::SelectionControlDelegate(QObject* parent) : QStyledItemDelegate(parent)
{
}

void SelectionControlDelegate::initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const
{
    QStyledItemDelegate::initStyleOption(option, index);
    const bool selected = option->state & QStyle::State_Selected;
    if (selected)
    {
        option->state = option->state & ~QStyle::State_Selected; // this will block selection-style = no highlight
    }
}



