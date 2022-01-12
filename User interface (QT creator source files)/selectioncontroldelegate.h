#ifndef SELECTIONCONTROLDELEGATE_H
#define SELECTIONCONTROLDELEGATE_H
#include <QStyledItemDelegate>


class SelectionControlDelegate: public QStyledItemDelegate
{
public:
    SelectionControlDelegate(QObject* parent = 0);
    void initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const override;
};

#endif // SELECTIONCONTROLDELEGATE_H
