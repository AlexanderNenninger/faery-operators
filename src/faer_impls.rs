use super::*;
use faer::modules::core::RealField;
use faer_entity::Entity;

impl<E: RealField> LinearOperator<E> for Matrix<DenseOwn<E>> {
    fn nrows(&self) -> usize {
        self.nrows()
    }

    fn ncols(&self) -> usize {
        self.ncols()
    }

    fn apply(&self, rhs: &Mat<E>) -> Mat<E> {
        self * rhs
    }
}

impl<E: RealField> Symmetric<E> for Matrix<DenseOwn<E>> {}
impl<E: RealField> PositiveDefinite<E> for Matrix<DenseOwn<E>> {}
