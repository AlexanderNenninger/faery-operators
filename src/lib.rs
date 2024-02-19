//! faery-operators explores abstract linear operators in faer

use faer::{
    modules::core::{inner::DenseOwn, Matrix, RealField},
    scale, Mat,
};

mod faer_impls;

/// Abstract Linear Operator that knows its size and can be applied to an element of the base space.
pub trait LinearOperator<E: RealField>
where
    Self: Sized,
{
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn apply(&self, rhs: &Mat<E>) -> Mat<E>;
}

pub trait Symmetric<E: RealField>
where
    Self: LinearOperator<E>,
{
    fn adjoint(self) -> Self {
        self
    }
}

// Marker Trait for Required for CG Algorithm
pub trait PositiveDefinite<E: RealField>
where
    Self: LinearOperator<E>,
{
}

impl<T, E: RealField> SymmetricPositiveDefinite<E> for T where T: Symmetric<E> + PositiveDefinite<E> {}

pub trait SymmetricPositiveDefinite<E: RealField>
where
    Self: Symmetric<E> + PositiveDefinite<E>,
{
    fn conjugate_gradient(self, rhs: Mat<E>, tol: E::Real, maxiter: usize) -> Matrix<DenseOwn<E>> {
        let mut x: Mat<E> = Mat::<E>::zeros(self.nrows(), 1);
        let mut residual = rhs - self.apply(&x);

        if residual.norm_max() <= tol {
            return x;
        }
        let mut p = residual.clone();
        let mut niter = 0;

        while niter < maxiter {
            let rtr: E::Real = (residual.as_ref().transpose() * residual.as_ref()).read(0, 0);
            let denominator: E::Real = (p.as_ref().transpose() * self.apply(&p)).read(0, 0);
            let alpha = scale(rtr.faer_div(denominator));

            x += alpha * p.as_ref();
            residual -= alpha * self.apply(&p);

            if residual.norm_max() <= tol {
                dbg!(niter);
                return x;
            }

            let beta = scale(
                (residual.as_ref().transpose() * residual.as_ref())
                    .read(0, 0)
                    .faer_div(rtr),
            );

            p = residual.as_ref() + beta * p;
            niter += 1;
        }
        dbg!("Exceeded iteration Limit");
        x
    }
}

#[cfg(test)]
mod tests {

    use faer::mat;

    use crate::SymmetricPositiveDefinite;

    #[test]
    fn it_does_not_work() {
        let mat = mat!([1.0, -3.0], [-3., 1.]);
        let b = mat!([1.0], [2.0]);

        let x = mat.clone().conjugate_gradient(b.clone(), 1e-10, 100);

        faer::assert_matrix_eq!(mat * &x, &b, comp = abs, tol = 1e-5);
    }
}
