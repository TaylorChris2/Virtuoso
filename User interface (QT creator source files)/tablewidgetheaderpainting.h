#ifndef TABLEWIDGETHEADERPAINTING_H
#define TABLEWIDGETHEADERPAINTING_H
#include <qheaderview.h>

class tablewidgetheaderpainting: public QHeaderView
{
public:
    tablewidgetheaderpainting();
protected:
  virtual void paintSection(QPainter *painter, const QRect &rect, int logicalIndex) const;

};

#endif // TABLEWIDGETHEADERPAINTING_H

