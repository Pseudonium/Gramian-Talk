from manim import *
import numpy as np
import itertools as it
from colour import Color

def rainbowify(textmobj):
    colors_of_the_rainbow = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE] 
    colors = it.cycle(colors_of_the_rainbow)
    for letter in textmobj:
        letter.set_color(next(colors))


class HeresWhereTheTalkReallyBegins(Scene):
    def construct(self):
        begin_text = Tex(r"Here's where the talk", r"\textbf{really} begins!", color=YELLOW).scale(2.5).arrange(DOWN)
        self.play(Write(begin_text))
        self.wait()

class FirstThingToAnimate(Scene):
    def construct(self):
        first_text = Tex(r"Matrices", r"$\Downarrow$", r"\textit{Linear Transformations}", color=YELLOW).scale(2).arrange(DOWN).shift(UP)
        self.play(Write(first_text))
        self.wait()

class LinearMapToMatrix(LinearTransformationScene):
    
    def __init__(self, **kwargs):
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False,
            x_min=-10,
            x_max=10,
            y_min=-10,
            y_max=10,
            **kwargs
        )
    
    
    def construct(self):
        matrix_1 = np.transpose([[1, 3], [-1, 4]])
        matrix_2 = np.transpose([[-1, 1], [2, 4]])
        matrix_3 = np.transpose([[-0.7, 0.7], [0, -0.9]])
        matrix_4 = np.transpose([[2, 1], [-1, 1]])
        
        self.wait()
        
        linear_text = Tex(r"Grid lines remain parallel and evenly spaced", color=YELLOW).to_edge(UP).add_background_rectangle()
        
        self.add(linear_text)
        
        self.play(Write(linear_text[1]))
        self.wait()
        
        self.play(ApplyMatrix(matrix_1, self.plane))
        self.wait()
        self.play(ApplyMatrix(np.linalg.inv(matrix_1), self.plane))
        self.wait()
        
        self.play(ApplyMatrix(matrix_2, self.plane))
        self.wait()
        self.play(ApplyMatrix(np.linalg.inv(matrix_2), self.plane))
        self.wait()
        
        self.play(ApplyMatrix(matrix_3, self.plane))
        self.wait()
        self.play(ApplyMatrix(np.linalg.inv(matrix_3), self.plane))
        self.wait()
        
        self.play(Rotate(self.plane, angle = 90 * DEGREES))
        self.wait()
        self.play(Rotate(self.plane, angle = -90 * DEGREES))
        self.wait()
        
        
        self.play(linear_text.animate.set_opacity(0))
        
        follow_text = Tex(r"Follow basis vectors").to_edge(UP).add_background_rectangle()
        
        i_hat = self.add_vector([1, 0, 0], color=GREEN)
        j_hat = self.add_vector([0, 1, 0], color=RED)
        
        self.wait()
        
        i_hat_label = self.get_vector_label(i_hat, MathTex(r"\hat{\imath}", color=GREEN))
        j_hat_label = self.get_vector_label(j_hat, MathTex(r"\hat{\jmath}", color=RED))
        self.play(Create(i_hat_label), Create(j_hat_label))
        self.wait()
        
        vector_v = self.add_vector([-2, -1, 0], color=YELLOW)
        
        v_label = self.get_vector_label(vector_v, MathTex(r"\vec{\mathbf{v} }", color=YELLOW))
        self.play(Create(v_label))
        self.wait()
        
        transform_i = self.get_vector([-2, 0, 0], color=GREEN)
        t_i_label = self.get_vector_label(transform_i, MathTex(r"-2\hat{\imath}", color=GREEN)).next_to(transform_i, UP)
        
        
        transform_j = self.get_vector([0, -1, 0], color=RED).shift([-2, 0, 0])
        t_j_label = self.get_vector_label(transform_j, MathTex(r"-\hat{\jmath}", color=RED)).shift([-2, 0, 0])
        
        self.play(TransformFromCopy(i_hat, transform_i), TransformFromCopy(j_hat, transform_j), TransformFromCopy(i_hat_label, t_i_label), TransformFromCopy(j_hat_label, t_j_label))
        self.wait()
        
        v_eq = MathTex(r"\vec{\mathbf{v} }", r"=", r"-2 \hat{\imath}", r"-\hat{\jmath}").scale(1.5).to_edge(UP).to_edge(LEFT)
        
        v_eq[0].set_color(YELLOW)
        v_eq[2].set_color(GREEN)
        v_eq[3].set_color(RED)
        
        v_eq.add_background_rectangle()
        
        self.add_foreground_mobject(v_eq)
        
        self.play(Write(v_eq[1:]))
        self.wait()
        
        v_eq_cpy = v_eq.copy()
        
        fade_grp = VGroup(v_label, i_hat_label, j_hat_label, t_i_label, t_j_label, transform_i, transform_j, v_eq)
        
        self.play(fade_grp.animate.set_opacity(0))
        
        
        self.apply_matrix(matrix_4)
        self.wait()
        
        self.add_foreground_mobject(v_eq_cpy)
        
        self.play(Write(v_eq_cpy[1:]))
        self.wait()
        
        i_hat_label_2 = self.get_vector_label(i_hat, MathTex(r"L(\hat{\imath})", color=GREEN))
        j_hat_label_2 = self.get_vector_label(j_hat, MathTex(r"L(\hat{\jmath})", color=RED))
        v_label_2 = self.get_vector_label(vector_v, MathTex(r"L(\vec{\mathbf{v} })", color=YELLOW))
        self.play(Create(i_hat_label_2), Create(j_hat_label_2), Create(v_label_2))
        self.wait()
        
        transform_i_2 = self.get_vector([-2, 0, 0], color=GREEN)
        t_i_label_2 = self.get_vector_label(transform_i, MathTex(r"-2L(\hat{\imath})", color=GREEN)).next_to(transform_i, UP).shift(2*RIGHT + UP)
        
        
        transform_j_2 = self.get_vector([0, -1, 0], color=RED).shift(-2 * np.array([2, 1, 0]))
        t_j_label_2 = self.get_vector_label(transform_j, MathTex(r"-L(\hat{\jmath})", color=RED)).shift(-2 * np.array([2, 1, 0]))
        
        self.play(TransformFromCopy(i_hat, transform_i_2), TransformFromCopy(j_hat, transform_j_2), TransformFromCopy(i_hat_label, t_i_label_2), TransformFromCopy(j_hat_label, t_j_label_2))
        self.wait()
        
        
        
        v_eq_2 = MathTex(r"L(\vec{\mathbf{v} })", r"=", r"-2 L(\hat{\imath})", r"-L(\hat{\jmath})").scale(1.5).next_to(v_eq_cpy, DOWN).to_edge(LEFT)
        
        v_eq_2[0].set_color(YELLOW)
        v_eq_2[2].set_color(GREEN)
        v_eq_2[3].set_color(RED)
        
        v_eq_2.add_background_rectangle()
        
        self.add_foreground_mobject(v_eq_2)
        self.play(Write(v_eq_2[1:]))
        self.wait()
        
        fade_grp_2 = VGroup(v_eq_cpy, v_eq_2, transform_i_2, t_i_label_2, transform_j_2, t_j_label_2, vector_v, v_label_2)
        self.play(fade_grp_2.animate.set_opacity(0))
        self.wait()
        
        determine_text = Tex(r"Transformation determined by basis vectors").to_edge(UP).add_background_rectangle()
        
        self.add(determine_text)
        
        self.play(Write(determine_text[1]))
        self.wait()
        
        i_hat_matrix = Matrix([[2], [1]], include_background_rectangle=True)
        i_hat_matrix[1].set_color(GREEN)
        i_hat_matrix.next_to(i_hat, UP + RIGHT)
        
        j_hat_matrix = Matrix([[-1], [1]], include_background_rectangle=True)
        j_hat_matrix[1].set_color(RED)
        j_hat_matrix.next_to(j_hat, UP + LEFT)
        
        self.play(Create(i_hat_matrix), Create(j_hat_matrix))
        self.wait()
        
        target_matrix = Matrix([[2, -1], [1, 1]], include_background_rectangle=True)
        target_matrix.set_column_colors(GREEN, RED)
        
        for entry in target_matrix.get_entries():
            entry.set_opacity(0)
        
        target_matrix.next_to(determine_text, DOWN).to_edge(LEFT)
        
        self.play(Create(target_matrix))
        self.wait()
        
        i_hat_matrix_cpy = i_hat_matrix.copy()
        
        self.play(i_hat_matrix_cpy.animate.move_to(target_matrix[1][0].get_center()).align_to(target_matrix, UP))
        self.wait()
        self.play(FadeOut(i_hat_matrix_cpy), target_matrix[1][0].animate.set_opacity(1), target_matrix[1][2].animate.set_opacity(1))
        self.wait()
        
        j_hat_matrix_cpy = j_hat_matrix.copy()
        
        self.play(j_hat_matrix_cpy.animate.move_to(target_matrix[1][1].get_center()).align_to(target_matrix, UP))
        self.wait()
        self.play(FadeOut(j_hat_matrix_cpy), target_matrix[1][1].animate.set_opacity(1), target_matrix[1][3].animate.set_opacity(1))
        self.wait()
        

class SecondThingToAnimate(Scene):
    def construct(self):
        second_text = Tex(r"Determinant", r"$\Downarrow$", r"\textit{Volume scale factor}").scale(2).arrange(DOWN).shift(UP)
        self.play(Write(second_text))
        self.wait()
        
        question_text = Tex(r"Why should it even exist?", color=YELLOW).scale(2).next_to(second_text, DOWN)
        
        self.play(Write(question_text))
        self.wait()
        
class WhyDeterminantShouldExist(LinearTransformationScene):
    def __init__(self, **kwargs):
        LinearTransformationScene.__init__(
            self,
            show_basis_vectors=False,
            x_min=-10,
            x_max=10,
            y_min=-10,
            y_max=10,
            include_background_plane=False,
            **kwargs
        )
    
    def add_square_custom(self, size, left_corner):
        square = Rectangle(
            color=YELLOW,
            width=size,
            height=size,
            stroke_color=YELLOW,
            stroke_width=3,
            fill_color=YELLOW,  
            fill_opacity=0.3,
        )
        square.move_to(self.plane.coords_to_point(*left_corner), DL)
        return square
    
    def approximate_circ_with_num(self, num):
        square_size = 2/num
        squares = []
        for left_edge in range(-1*num, 1*num):
            for bottom_edge in range(-1*num, 1*num):
                # Check if in circle or not
                center_x = (left_edge + 0.5) * square_size
                center_y = (bottom_edge + 0.5) * square_size
                if (center_x**2 + center_y**2) <= 1:
                    squares.append(self.add_square_custom(square_size, [left_edge * square_size, bottom_edge * square_size]))
        return VGroup(*squares)
    
    def construct(self):
        matrix = np.transpose([[2, 1], [-1, 1]])
        self.add_unit_square()
        square_1 = self.square
        
        self.apply_matrix(matrix)
        self.wait()
        
        scale_text = Tex(r"Scaled by $k$").to_edge(UP).to_edge(RIGHT).add_background_rectangle()
        scale_arrow = Arrow(scale_text.get_corner(DL), square_1.get_center())
        self.add(scale_text)
        self.play(Write(scale_text[1:]), Create(scale_arrow))
        self.wait()
        
        down_implies = MathTex(r"\Downarrow").next_to(scale_text, DOWN)
        
        all_text = Tex(r"All scaled by $k$", color=YELLOW).next_to(down_implies, DOWN).to_edge(RIGHT).add_background_rectangle()
        
        self.add(all_text)
        self.play(Create(down_implies), Write(all_text[1:]))
        self.wait()
        fade_grp = VGroup(all_text, down_implies, scale_text, scale_arrow)
        self.play(fade_grp.animate.set_opacity(0))
        
        self.apply_matrix(np.linalg.inv(matrix))
        self.wait()
        
        square_2 = self.add_square_custom(1, [-2, 1])
        square_3 = self.add_square_custom(0.5, [2, 0])
        square_4 = self.add_square_custom(2, [0, -3])
        square_5 = self.add_square_custom(0.2, [-1, 0])
        self.play(DrawBorderThenFill(square_2), DrawBorderThenFill(square_3), DrawBorderThenFill(square_4), DrawBorderThenFill(square_5))
        self.wait()
        self.moving_mobjects = []
        self.add_transformable_mobject(square_2)
        self.add_transformable_mobject(square_3)
        self.add_transformable_mobject(square_4)
        self.add_transformable_mobject(square_5)
        
        self.apply_matrix(matrix)
        
        square_text = Tex(r"All grid squares scaled by $k$").to_edge(UP).add_background_rectangle()
        self.add(square_text)
        self.play(Write(square_text[1:]))
        self.wait()
        self.play(square_text.animate.set_opacity(0))
        self.wait()
        self.moving_mobjects = []
        self.apply_matrix(np.linalg.inv(matrix))
        fade_grp_2 = VGroup(self.square, square_2, square_3, square_4, square_5)
        self.play(fade_grp_2.animate.set_opacity(0))
        
        test_circ = Circle(fill_color=RED,  
            fill_opacity=0.3,)
        self.play(Create(test_circ))
        self.add_transformable_mobject(test_circ)
        
        squares_1 = self.approximate_circ_with_num(5)
        self.play(Create(squares_1))
        self.play(FadeOut(squares_1))
        squares_2 = self.approximate_circ_with_num(10)
        self.play(Create(squares_2))
        self.play(FadeOut(squares_2))
        
        squares_3 = self.approximate_circ_with_num(20)
        self.play(Create(squares_3))
        self.add_transformable_mobject(squares_3)
        self.wait()
        self.moving_mobjects = []
        self.apply_matrix(matrix)
        self.wait()
        
        arb_text = Tex(r"Arbitrary region scaled by $k$", color=YELLOW).to_edge(UP).add_background_rectangle()
        self.add(arb_text)
        self.play(Write(arb_text[1:]))
        self.wait()
        

class DeterminantToMeasureArea(Scene):
    def construct(self):
        title_text = Tex(r"What we want").scale(2).to_edge(UP)
        self.play(Write(title_text))
        self.wait()
        interested_text_0 = Tex(r"Interested in determining area of parallelogram,")
        interested_text_1 = Tex(r"or 3D volume of parallelepiped,")
        interested_text_2 = Tex(r"or general volume of parallelotope")
        
        text_grp = VGroup(interested_text_0, interested_text_1, interested_text_2)
        
        text_grp.to_edge(UP).arrange(DOWN)
        
        self.play(Write(interested_text_0))
        self.wait()
        self.play(Write(interested_text_1))
        self.wait()
        self.play(Write(interested_text_2))
        self.wait()
        
        what_text = Tex(r"What do we already know?", color=YELLOW).next_to(interested_text_2, DOWN)
        
        self.play(Write(what_text))
        self.wait()

class WhatWeCanDoSoFar0(Scene):
    def construct(self):
        title_text = Tex("What we know so far").scale(2)
        title_text.to_edge(UP)
        
        self.play(Write(title_text))
        
        self.wait(2)
    
        determinant_text = Tex(r"Given $n$ vectors in $\mathbb{R}^n$ $\longrightarrow$ ", "take the determinant")
        determinant_text[1].set_color(GREEN)
        
        self.play(Write(determinant_text))
        
        self.wait(2)
        
        self.play(determinant_text.animate.shift(UP))
        
        cross_text = Tex(r"Given 2 vectors in $\mathbb{R}^3$ $\longrightarrow$ ", "magnitude of cross product")
        cross_text[1].set_color(GREEN)
        self.play(Write(cross_text))
        
        self.wait(4)
        
        length_text = Tex(r"Given 1 vector in $\mathbb{R}^n$ $\longrightarrow$ ", "find its length").shift(DOWN)
        length_text[1].set_color(GREEN)
        
        self.play(Write(length_text))
        
        self.wait(3)
        
        box = SurroundingRectangle(determinant_text)
        self.play(Create(box))
        self.wait()


class ParallelogramArea(LinearTransformationScene):
    def construct(self):
        matrix = [[2, 2], [0, 3]]
        m0 = Matrix(matrix, include_background_rectangle = True).shift(4.5*LEFT + 2*UP)
        m0.set_column_colors("#83C167", "#FC6255")
        
        self.add_unit_square()
        self.add_foreground_mobject(m0)
        self.wait()
        self.moving_mobjects = []
        self.apply_matrix(matrix)
        bottom_brace = Brace(self.i_hat, DOWN)
        right_brace = Brace(self.square, RIGHT)
        width = Tex(str(2))
        height = Tex(str(3))
        width.next_to(bottom_brace, DOWN)
        height.next_to(right_brace, RIGHT)
        for mob in bottom_brace, width, right_brace, height:
            mob.add_background_rectangle()
            self.play(Create(mob, run_time = 0.5))
        det = get_det_text(m0,
                    determinant=6,
                    initial_scale_factor=1)
        self.play(Create(det))
        self.wait()


class ParallelepipedVolume(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(z_min=-4.5, z_max=4.5).add_coordinates()
        self.add(axes)
        array_u = [2, 0, 1]
        array_v = [0, 2, 1]
        array_w = [0, 0, 1]
        matrix = [array_u, array_v, array_w]
        
        m0 = Matrix(matrix)
        m0.set_column_colors(GREEN, RED, BLUE)
        
        det_text = get_det_text(m0, determinant=4, initial_scale_factor=1)
        det_grp = VGroup(m0, det_text)
        det_grp.to_edge(UP).to_edge(LEFT)
        
        i_hat = Arrow3D([0, 0, 0], [1, 0, 0], color=GREEN)
        j_hat = Arrow3D([0, 0, 0], [0, 1, 0], color=RED)
        k_hat = Arrow3D([0, 0, 0], [0, 0, 1], color=BLUE)
        
        self.set_camera_orientation(phi = 60 * DEGREES, theta = -100 * DEGREES)
        
        cube = Cube(side_length=1).move_to([0.5, 0.5, 0.5])
        
        vectors = VGroup(i_hat, j_hat, k_hat)
        self.add(vectors)
        self.play(Create(cube))
        self.add_fixed_in_frame_mobjects(m0)
        self.play(Create(m0))
        self.wait()
        self.play(FadeOut(vectors))
        self.play(ApplyMatrix(matrix, cube))
        self.wait()
        self.move_camera(phi=90 * DEGREES, theta = -90 * DEGREES)
        self.add_fixed_in_frame_mobjects(det_text)
        self.play(Write(det_text))
        self.wait()
        

class WhatWeCanDoSoFar1(Scene):
    def construct(self):
        title_text = Tex("What we know so far").scale(2)
        title_text.to_edge(UP)
        
        self.add(title_text)
    
        determinant_text = Tex(r"Given $n$ vectors in $\mathbb{R}^n$ $\longrightarrow$ ", "take the determinant").shift(UP)
        determinant_text[1].set_color(GREEN)
        
        self.add(determinant_text)
        
        cross_text = Tex(r"Given 2 vectors in $\mathbb{R}^3$ $\longrightarrow$ ", "magnitude of cross product")
        cross_text[1].set_color(GREEN)
        self.add(cross_text)
        
        length_text = Tex(r"Given 1 vector in $\mathbb{R}^n$ $\longrightarrow$ ", "find its length").shift(DOWN)
        length_text[1].set_color(GREEN)
        
        self.add(length_text)
        
        box = SurroundingRectangle(cross_text)
        self.play(Create(box))
        self.wait()


class ParallelogramAreaIn3D(ThreeDScene):
    def construct(self):
    
        axes = ThreeDAxes(z_min=-4.5, z_max=4.5)
        
        vector_v = Arrow3D(np.array([0, 0, 0]), np.array([1, -2, 1]), color="#83C167")
        vector_w = Arrow3D(np.array([0, 0, 0]), np.array([2, 0, 2]), buff=0, color="#FC6255")
        vector_cross = Arrow3D([0,0,0], [-4,0,4], color=BLUE)
        
        v_label = MathTex(r"\vec{\mathbf{v} }", color="#83C167")
        w_label = MathTex(r"\vec{\mathbf{w} }", color="#FC6255")
        cross_label = MathTex(r"\vec{\mathbf{v} } \times \vec{\mathbf{w} }", color=BLUE)
        
        parallelogram = Polygon([0, 0, 0], [2, 0, 2], [3, -2, 3], [1, -2, 1], color="white")
        parallelogram.set_fill(WHITE, opacity=0.5)
        
        mobjects = [axes, vector_v, vector_w, vector_cross, parallelogram]
        for mobj in mobjects:
            mobj = mobj.scale(0.7, about_point=[0,0,0])
        
        self.add(vector_v)
        self.add(vector_w)
        self.add(axes)
        
        self.set_camera_orientation(phi=60 * DEGREES, theta=-100 * DEGREES)
        
        v_label.next_to(vector_v, RIGHT*2 + UP*0)
        v_label.shift(UP*0.4)
        w_label.next_to(vector_w, UP*3)
        
        
        
        self.add_fixed_in_frame_mobjects(v_label, w_label)
        
        self.wait()
        
        self.play(Create(parallelogram))
        
        question_text = Tex(r"Area of parallelogram ", r"$= |\vec{\mathbf{v} } \times \vec{\mathbf{w} }|$")
        question_text[1].set_opacity(0)
        
        question_text[0].set_color(YELLOW)
        question_text[1].set_color(BLUE)
        
        question_text.to_edge(UP).to_edge(RIGHT)
        
        self.add_fixed_in_frame_mobjects(question_text)
        self.play(Write(question_text))
        
        self.wait()
        
        self.add(vector_cross)
        
        cross_label.next_to(vector_cross, UP*6)
        self.add_fixed_in_frame_mobjects(cross_label)
        
        self.wait()
        
        self.play(question_text[1].animate.set_opacity(1))
        
        self.wait()


class ParallelogramAreaIn3DAlgebra(Scene):
    def construct(self):
        coord_text = MathTex(r"\vec{\mathbf{v} } = \begin{bmatrix} 1 \\ -2 \\ 1 \end{bmatrix} \space", r"\vec{\mathbf{w} } = \begin{bmatrix} 2 \\ 0 \\ 2 \end{bmatrix}")
        coord_text[0].set_color(GREEN)
        coord_text[1].set_color(RED)
        
        coord_text.shift(UP*2)
        
        self.play(Write(coord_text))
        
        cross_text = MathTex(r"\vec{\mathbf{v} }", r"\times", r"\vec{\mathbf{w} }", r"=", r"\begin{bmatrix} -4 \\ 0 \\ 4 \end{bmatrix}")
        cross_text[0].set_color(GREEN)
        cross_text[2].set_color(RED)
        cross_text[4].set_color(BLUE)
        
        self.play(Write(cross_text))
        self.wait()
        
        mag_text = MathTex(r"|", r"\vec{\mathbf{v} }", r"\times", r"\vec{\mathbf{w} }", r"| =", r"\sqrt{32} \approx 5.66")
        mag_text[1].set_color(GREEN)
        mag_text[3].set_color(RED)
        mag_text[5].set_color(BLUE)
        mag_text.next_to(cross_text, DOWN)
        self.play(Write(mag_text))
        self.wait()


class WhatWeCanDoSoFar2(Scene):
    def construct(self):
        title_text = Tex("What we know so far").scale(2)
        title_text.to_edge(UP)
        
        self.add(title_text)
    
        determinant_text = Tex(r"Given $n$ vectors in $\mathbb{R}^n$ $\longrightarrow$ ", "take the determinant").shift(UP)
        determinant_text[1].set_color(GREEN)
        
        self.add(determinant_text)
        
        cross_text = Tex(r"Given 2 vectors in $\mathbb{R}^3$ $\longrightarrow$ ", "magnitude of cross product")
        cross_text[1].set_color(GREEN)
        self.add(cross_text)
        
        length_text = Tex(r"Given 1 vector in $\mathbb{R}^n$ $\longrightarrow$ ", "find its length").shift(DOWN)
        length_text[1].set_color(GREEN)
        
        self.add(length_text)
        
        question_text = Tex(r"What about the general case?", color=YELLOW).scale(2)
        question_text.next_to(length_text, DOWN)
        
        self.play(Write(question_text))
        
        
        self.wait()

class TheGeneralCase(Scene):
    def construct(self):
        title_text = Tex(r"Problem statement").scale(2).to_edge(UP)
        
        self.play(Write(title_text))
        self.wait()
        
        given_text = Tex(r"Given $n$ vectors ", r"$\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}$", r" in $\mathbb{R}^d$").next_to(title_text, DOWN).shift(DOWN*0.5)
        given_text[1].set_color(BLUE)
        
        self.play(Write(given_text))
        self.wait()
        
        want_text = Tex(r"Want to compute ", r"volume of $n$-dimensional parallelotope").next_to(given_text, DOWN)
        want_text[1].set_color(YELLOW)
        
        self.play(Write(want_text))
        self.wait()
        
        note_text = Tex(r"Assume $n < d$ - if equal use determinant, if greater then just $0$").next_to(want_text, DOWN)
        self.play(Write(note_text))
        self.wait()
        self.play(FadeOut(note_text))
        self.wait()
        
        hard_text = Tex(r"Hard problem - ", r"can't just take determinant", color=RED).next_to(want_text, DOWN)
        
        self.play(Write(hard_text))
        self.wait()
        
        example_matrix = Matrix([[1, 2], [-2, 0], [1, 2]]).next_to(hard_text, DOWN)
        example_matrix.set_column_colors(GREEN, RED)
        det_text = get_det_text(example_matrix, r"???", initial_scale_factor=1)
        
        self.play(Create(example_matrix))
        self.wait()
        self.play(Write(det_text))
        self.wait()
        
        self.play(FadeOut(example_matrix), FadeOut(det_text))
        self.wait()
        
        illus_text = Tex(r"Simpler problem - ", r"other way to do for 2 3D vectors?").next_to(hard_text, DOWN)
        illus_text[1].set_color(YELLOW)
        self.play(Write(illus_text))
        self.wait()
        


class RotateDownTo2D(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(z_min=-4.5, z_max=4.5).add_coordinates()
        
        vector_v = [1, -2, 1]
        vector_w = [2, 0, 2]
        
        arrow_v = Arrow3D(np.array([0, 0, 0]), np.array([1, -2, 1]), color="#83C167")
        arrow_w = Arrow3D(np.array([0, 0, 0]), np.array([2, 0, 2]), buff=0, color="#FC6255")
        
        parallelogram = Polygon([0, 0, 0], [2, 0, 2], [3, -2, 3], [1, -2, 1], color="white")
        parallelogram.set_fill(WHITE, opacity=0.5)
        
        mobjects = [axes, arrow_v, arrow_w, parallelogram]
        for mobj in mobjects:
            mobj = mobj.scale(0.7, about_point=[0,0,0])
        
        self.add(arrow_v)
        self.add(arrow_w)
        self.add(axes)
        
        self.set_camera_orientation(phi=60 * DEGREES, theta=-100 * DEGREES)
        
        self.play(Create(parallelogram))
        
        known_text = Tex(r"Know: Area of parallelogram ", r"$= |\vec{\mathbf{v} } \times \vec{\mathbf{w} }|$")
        
        known_text[0].set_color(YELLOW)
        known_text[1].set_color(BLUE)
        
        
        question_text = Tex(r"How else could we find the area?", color=YELLOW)
        
        idea_text = Tex(r"Idea: Rotate into a lower dimension", color=BLUE)
        
        coord_text = MathTex(r"\vec{\mathbf{v} } = \begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix} \space", r"\vec{\mathbf{w} } = \begin{pmatrix} 2 \\ 0 \\ 2 \end{pmatrix}")
        
        coord_text[0].set_color("#83C167")
        coord_text[1].set_color("#FC6255")
        
        magic_text = Tex("Find a rotation matrix... ", "by magic")
        rainbowify(magic_text[1])
        magic_text[1].set_opacity(0)
        
        mid_axis = normalize([0, 0, 1] + get_unit_normal(vector_v, vector_w))
        
        matrix = DecimalMatrix(rotation_matrix(180 * DEGREES, mid_axis), element_to_mobject_config={'num_decimal_places': 2}).scale(0.8)
        
        apply_text = Tex("Apply the matrix")
        
        perspective_text = Tex("Change our perspective", color=YELLOW)
        
        known_text.to_edge(UP).to_edge(RIGHT)
        question_text.to_edge(UP).to_edge(RIGHT)
        idea_text.to_edge(UP).to_edge(RIGHT)
        coord_text.to_edge(UP).to_edge(RIGHT)
        magic_text.to_edge(UP).to_edge(RIGHT)
        matrix.to_edge(UP).to_edge(RIGHT)
        apply_text.to_edge(UP).to_edge(RIGHT)
        perspective_text.to_edge(UP).to_edge(RIGHT)
        
        self.wait(2)
        
        self.add_fixed_in_frame_mobjects(known_text)
        
        self.play(Write(known_text))
        self.wait(3)
        self.play(FadeOut(known_text))
        
        self.add_fixed_in_frame_mobjects(question_text)
        
        self.play(Write(question_text))
        self.wait(2)
        self.play(FadeOut(question_text))
        
        self.add_fixed_in_frame_mobjects(idea_text)
        
        self.play(Write(idea_text))
        self.wait(2)
        self.play(FadeOut(idea_text))
        
        self.add_fixed_in_frame_mobjects(coord_text)
        
        self.play(Write(coord_text))
        self.wait(2)
        self.play(FadeOut(coord_text))
        
        self.add_fixed_in_frame_mobjects(magic_text)
        
        self.play(Write(magic_text))
        self.wait(3)
        self.play(magic_text[1].animate.set_opacity(1))
        self.wait(3)
        self.play(FadeOut(magic_text))
        magic_text.set_opacity(0)
        magic_text[1].set_opacity(0)
        
        self.add_fixed_in_frame_mobjects(matrix)
        
        self.play(Create(matrix))
        self.wait(2)
        self.play(FadeOut(matrix))
        
        self.add_fixed_in_frame_mobjects(apply_text)
        
        self.play(Write(apply_text))
        self.wait(2)
        
        self.play(Rotate(arrow_v, axis=mid_axis, about_point=[0,0,0]), Rotate(arrow_w, axis=mid_axis, about_point=[0,0,0]), Rotate(parallelogram, axis=mid_axis, about_point=[0,0,0]))
        
        self.play(FadeOut(apply_text))
        
        self.add_fixed_in_frame_mobjects(perspective_text)
        
        self.play(Write(perspective_text))
        self.wait(3)
        
        self.move_camera(phi=0, gamma=0, theta=-90 * DEGREES)
        
        self.wait(2)
        self.play(FadeOut(perspective_text))
        
        result = Tex(r"Now regard as 2D vectors", color=BLUE)
        
        result.to_edge(UP).to_edge(RIGHT)
        
        self.play(Write(result))
        
        self.wait()



class RotateDownTo2DAlgebra(Scene):
    def construct(self):
        array_v = [1, -2, 1]
        array_w = [2, 0, 2]
        
        coord_text = MathTex(r"\vec{\mathbf{v} } = \begin{bmatrix} 1 \\ -2 \\ 1 \end{bmatrix} \space", r"\vec{\mathbf{w} } = \begin{bmatrix} 2 \\ 0 \\ 2 \end{bmatrix}")
        
        coord_text[0].set_color(GREEN)
        coord_text[1].set_color(RED)
        
        coord_text.to_edge(UP)
        
        self.play(Write(coord_text))
        self.wait()
        
        mid_axis = normalize([0, 0, 1] + get_unit_normal(array_v, array_w))
        
        matrix = DecimalMatrix(rotation_matrix(180 * DEGREES, mid_axis), element_to_mobject_config={'num_decimal_places': 2}).scale(0.8).to_edge(UP).to_edge(LEFT)
        
        self.play(Create(matrix))
        self.wait()
        
        target_matrix = np.matmul(rotation_matrix(180 * DEGREES, mid_axis), np.transpose([array_v, array_w]))[:2]
        
        target_matrix = DecimalMatrix(target_matrix, element_to_mobject_config={'num_decimal_places': 2}).scale(0.8).to_edge(UP).to_edge(RIGHT)
        
        target_matrix.set_column_colors(GREEN, RED)
        
        for i in range(2):
            for j in range(2):
                target_matrix[0][i + 2*j].set_opacity(0)
        
        self.play(Create(target_matrix))
        self.wait()
        
        matrix_1 = matrix.copy().next_to(matrix, DOWN)
        
        start_v = DecimalMatrix([[elem] for elem in array_v], element_to_mobject_config={'num_decimal_places': 2}).scale(0.8).next_to(matrix_1, RIGHT).set_color(GREEN)
        
        equals = MathTex(r"=", color=BLUE).next_to(start_v, RIGHT)
        
        result_v = DecimalMatrix([[elem] for elem in np.matmul(rotation_matrix(180 * DEGREES, mid_axis), array_v)], element_to_mobject_config={'num_decimal_places': 2}).scale(0.8).next_to(equals, RIGHT).set_color(GREEN)
        
        application_1 = VGroup(matrix_1, start_v, equals, result_v)
        
        
        self.play(Create(application_1))
        self.wait()
        
        temp = result_v.copy()
        
        self.play(temp.animate.move_to(target_matrix[0][0].get_center()).align_to(target_matrix, UP))
        self.wait()
        
        self.play(FadeOut(temp), target_matrix[0][0].animate.set_opacity(1), target_matrix[0][2].animate.set_opacity(1))
        self.wait()
        
        start_w = DecimalMatrix([[elem] for elem in array_w], element_to_mobject_config={'num_decimal_places': 2}).scale(0.8).next_to(matrix_1, RIGHT).set_color(RED)
        
        result_w = DecimalMatrix([[elem] for elem in np.matmul(rotation_matrix(180 * DEGREES, mid_axis), array_w)], element_to_mobject_config={'num_decimal_places': 2}).scale(0.8).next_to(equals, RIGHT).set_color(RED)
        
        self.play(FadeOut(start_v), FadeOut(result_v), FadeIn(start_w), FadeIn(result_w))
        self.wait()
        
        temp_2 = result_w.copy()
        
        self.play(temp_2.animate.move_to(target_matrix[0][1].get_center()).align_to(target_matrix, UP))
        self.wait()
        
        self.play(FadeOut(temp_2), target_matrix[0][1].animate.set_opacity(1), target_matrix[0][3].animate.set_opacity(1))
        self.wait()
        
        self.play(FadeOut(matrix_1), FadeOut(start_w), FadeOut(equals), FadeOut(result_w))
        self.wait()
        self.play(target_matrix.animate.move_to([0, 0, 0]))
        self.wait()
        
        det_text = get_det_text(target_matrix, determinant=5.66, initial_scale_factor=1)
        self.play(Write(det_text))
        self.wait()
      


class MagicIsBad(Scene):
    def construct(self):
        big_text = Tex(r"Magic").scale(4)
        rainbowify(big_text[0])
        self.play(Write(big_text))
        self.wait()
        
        cross_0 = Line(big_text.get_corner(DL), big_text.get_corner(UR), color=RED, stroke_width=20)
        cross_1 = Line(big_text.get_corner(UL), big_text.get_corner(DR), color=RED, stroke_width=20)
        
        self.play(Create(cross_0), Create(cross_1))
        self.play(FadeOut(big_text), FadeOut(cross_0), FadeOut(cross_1))
        math_text = Tex(r"Mathematics", color=YELLOW).scale(4)
        self.play(Write(math_text))
        self.wait()



class IdeaChangeBetweenOrthonormalBases(Scene):
    def construct(self):
        idea_text = Tex(r"Instead of rotation, can use orthogonal matrix").to_edge(UP).shift(DOWN*1.5)
        self.play(Write(idea_text))
        self.wait()
        
        recap_text = Tex(r"Orthogonal matrix $\Leftrightarrow$ ", r"$\mathbf{P}^T \mathbf{P} = \mathbf{I}$").next_to(idea_text, DOWN)
        recap_text[1].set_color(PURPLE)
        
        self.play(Write(recap_text))
        self.wait()
        
        might_text = Tex(r"(Might flip area - doesn't affect size though)").next_to(recap_text, DOWN)
        
        self.play(Write(might_text))
        self.wait()
        
        idea_text_2 = Tex(r"Orthogonal matrix $\Leftrightarrow$ ", r"Change between orthonormal bases").next_to(might_text, DOWN)
        idea_text_2[1].set_color(YELLOW)
        self.play(Write(idea_text_2))
        self.wait()


class ChangeBetweenOrthonormalBases(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(z_min=-4.5, z_max=4.5)
        
        array_v = [1, -2, 1]
        array_w = [2, 0, 2]
        
        mid_axis = normalize([0, 0, 1] + get_unit_normal(array_v, array_w))
        rotate_matrix = rotation_matrix(180 * DEGREES, mid_axis)
        
        array_1 = [1, 0, 0]
        array_2 = [0, 1, 0]
        array_3 = [0, 0, 1]
        
        array_4 = np.transpose(np.matmul(rotate_matrix, np.transpose(array_1)))
        array_5 = np.transpose(np.matmul(rotate_matrix, np.transpose(array_2)))
        array_6 = np.transpose(np.matmul(rotate_matrix, np.transpose(array_3)))
        
        arrays = [array_1, array_2, array_3, array_4, array_5, array_6]
        vectors = list()
        
        for array, color in zip(arrays, [BLUE, BLUE, BLUE, PURPLE, PURPLE, PURPLE]):
            vectors.append(Arrow3D([0, 0, 0], array, color=color))
        
        vectors = VGroup(*vectors)
        
        self.set_camera_orientation(phi=60 * DEGREES, theta=-100 * DEGREES)
        
        self.add(axes)
        self.play(Create(vectors))
        #self.add(vectors)
        #self.begin_ambient_camera_rotation(rate=0.2)
        self.wait()
        
        question_text = Tex(r"How do we know change of basis matrix is orthogonal?").to_edge(UP).add_background_rectangle()
        answer_text = Tex(r"Preserves dot product", color=YELLOW).next_to(question_text, DOWN).add_background_rectangle()
        
        self.add_fixed_in_frame_mobjects(question_text)
        self.play(Write(question_text[1]))
        self.wait()
        
        self.play(Rotate(vectors[3:], axis=mid_axis, about_point=[0,0,0]))
        self.wait()
        
        self.add_fixed_in_frame_mobjects(answer_text)
        self.play(Write(answer_text[1]))
        self.wait()
        

class ChangeBetweenOrthonormalBasesAlgebra(Scene):
    def construct(self):
        linear_text = Tex(r"$\mathbf{P}$", r" sends ", r"$\{\vec{\mathbf{e} }_i\}$", r" to ", r"$\{\vec{\mathbf{e} }_i'\}$").to_edge(UP).to_edge(LEFT)
        linear_text[0].set_color(PURPLE)
        linear_text[2].set_color(BLUE)
        linear_text[4].set_color(GREEN)
        
        line_1 = MathTex(r"\vec{\mathbf{e} }_i", r" \cdot ", r"\vec{\mathbf{e} }_j", r" = \delta_{ij}").next_to(linear_text, DOWN).to_edge(LEFT)
        line_1[0].set_color(BLUE)
        line_1[2].set_color(BLUE)
        
        line_2 = MathTex(r"(", r"\mathbf{P}", r" \vec{\mathbf{e} }_i", r") \cdot (", r"\mathbf{P}", r"\vec{\mathbf{e} }_j", r") = ", r"\vec{\mathbf{e} }_i'", r"\cdot", r"\vec{\mathbf{e} }_j'", r"= \delta_{ij}").next_to(line_1, DOWN).to_edge(LEFT)
        line_2[1].set_color(PURPLE)
        line_2[2].set_color(BLUE)
        line_2[4].set_color(PURPLE)
        line_2[5].set_color(BLUE)
        line_2[7].set_color(GREEN)
        line_2[9].set_color(GREEN)
        
        line_3 = Tex(r"$\mathbf{P}$", r" preserves dot product between basis vectors").next_to(line_2, DOWN).to_edge(LEFT)
        line_3[0].set_color(PURPLE)
        line_3[1].set_color(YELLOW)
        
        line_4 = Tex(r"$\Rightarrow$", r"$\mathbf{P}$", r" preserves dot product in general").next_to(line_3, DOWN).to_edge(LEFT)
        line_4[1].set_color(PURPLE)
        line_4[2].set_color(YELLOW)
        
        self.play(Write(linear_text))
        self.wait()
        self.play(Write(line_1))
        self.wait()
        self.play(Write(line_2))
        self.wait()
        self.play(Write(line_3))
        self.wait()
        self.play(Write(line_4))
        self.wait()


class GeneralStrategy0(Scene):
    
    @staticmethod
    def bottom_mid(mobj):
        return (mobj.get_corner(DL) + mobj.get_corner(DR))/2
    
    @staticmethod
    def top_mid(mobj):
        return (mobj.get_corner(UL) + mobj.get_corner(UR))/2
    
    def construct(self):
        line_1 = Tex(r"Given $n$ vectors ", r"$\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}$", r" in $\mathbb{R}^d$").to_edge(UP)
        line_1[1].set_color(BLUE)
        
        line_2 = Tex(r"Get a ", r"basis", r" for ", r"$span\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}$").next_to(line_1, DOWN).shift(DOWN*0.5)
        line_2[1].set_color(YELLOW)
        line_2[3].set_color(BLUE)
        
        arrow_1_2 = Arrow(self.bottom_mid(line_1), self.top_mid(line_2))
        
        line_3 = Tex(r"Convert to ", r"an orthonormal basis").next_to(line_2, DOWN).shift(DOWN*0.5)
        line_3[1].set_color(YELLOW)
        
        arrow_2_3 = Arrow(self.bottom_mid(line_2), self.top_mid(line_3))
        
        line_4 = Tex(r"Extend to ", r"an orthonormal basis of $\mathbb{R}^d$").next_to(line_3, DOWN).shift(DOWN*0.5)
        line_4[1].set_color(YELLOW)
        
        arrow_3_4 = Arrow(self.bottom_mid(line_3), self.top_mid(line_4))
        
        line_5 = Tex(r"Use change of basis matrix from this to ", r"standard basis").next_to(line_4, DOWN).shift(DOWN*0.5)
        line_5[1].set_color(YELLOW)
        
        arrow_4_5 = Arrow(self.bottom_mid(line_4), self.top_mid(line_5))
        
        self.play(Write(line_1))
        self.wait()
        self.play(Write(line_2))
        self.wait()
        self.play(Write(line_3))
        self.wait()
        self.play(Write(line_4))
        self.wait()
        self.play(Write(line_5))
        self.wait()
        
        standard_brace = Brace(line_5[1])
        standard_brace_text = standard_brace.get_tex(r"\{\begin{bmatrix} 1 \\ 0 \\ \dots \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \\ \dots \\ 0 \end{bmatrix}, \dots, \begin{bmatrix} 0 \\ 0 \\ \dots \\ 1 \end{bmatrix}\}").scale(0.5).next_to(standard_brace, DOWN)
        
        self.play(Create(standard_brace), Write(standard_brace_text))
        self.wait()
        self.play(FadeOut(standard_brace), FadeOut(standard_brace_text))
        self.wait()
        
        box = SurroundingRectangle(line_2)
        self.play(Create(box))
        self.wait()
       


class BasisAsSubset(VectorScene):
    def construct(self):
        self.plane = self.add_plane()
        
        array_u = np.array([1, 0, 0])
        array_v = np.array([0, 1, 0])
        array_w = np.array([2, 1, 0])
        
        array_z = [-2, 3, 0]
        
        vector_u = self.add_vector(array_u, color=BLUE)
        u_label = MathTex(r"\vec{\mathbf{u} }", color=BLUE)
        u_label.next_to(vector_u, DOWN*0.3 + RIGHT*0.3)
        
        vector_v = self.add_vector(array_v, color=GREEN)
        v_label = MathTex(r"\vec{\mathbf{v} }", color=GREEN)
        v_label.next_to(vector_v, UP*0.3 + LEFT*0.3)
        
        
        vector_w = self.add_vector(array_w, color=RED)
        w_label = MathTex(r"\vec{\mathbf{w} }", color=RED)
        w_label.next_to(vector_w, UP*0.3 + RIGHT*0.3)
        
        self.play(Write(u_label), Write(v_label), Write(w_label))
        self.wait()
        
        span_set = Tex(r"$\{$", r"$\vec{\mathbf{u} },$",  r"$\vec{\mathbf{v} },$",  r"$\vec{\mathbf{w} }$", r"$\}$ is a ", r"\textit{spanning set}", " for $\mathbb{R}^2$")
        span_set[1].set_color(BLUE)
        span_set[2].set_color(GREEN)
        span_set[3].set_color(RED)
        span_set[5].set_color(YELLOW)
        
        span_set.add_background_rectangle()
        
        span_set.to_edge(UP).to_edge(RIGHT).shift(UP*0.5)
        self.play(Create(span_set))
        self.wait()
        
        vector_z = self.add_vector(array_z, color=YELLOW)
        z_label = MathTex(r"\vec{\mathbf{z} }", color=YELLOW)
        z_label.next_to(vector_z, UP*0.3 + LEFT*0.3)
        
        self.play(Write(z_label))
        
        transform_u = self.get_vector(-4 * array_u, color=BLUE)
        t_u_label = MathTex(r"-4\vec{\mathbf{u} }", color=BLUE)
        t_u_label.next_to(transform_u, DOWN*0.3 + LEFT*0.3)
        
        transform_v = self.get_vector(2 * array_v, color=GREEN).shift([-4, 0, 0])
        t_v_label = MathTex(r"2\vec{\mathbf{v} }", color=GREEN)
        t_v_label.next_to(transform_v, UP*0.3 + LEFT*0.3)
        
        transform_w = self.get_vector(array_w, color=RED).shift([-4, 2, 0])
        t_w_label = MathTex(r"\vec{\mathbf{w} }", color=RED)
        t_w_label.shift(transform_w.get_end() + LEFT + DOWN*0.2)
        
        self.play(
            TransformFromCopy(vector_u, transform_u),
            TransformFromCopy(u_label, t_u_label),
            TransformFromCopy(vector_v, transform_v),
            TransformFromCopy(v_label, t_v_label),
            TransformFromCopy(vector_w, transform_w),
            TransformFromCopy(w_label, t_w_label)
        )
        
        z_eq_1 = MathTex(r"\vec{\mathbf{z} }", r"=", r"-4\vec{\mathbf{u} }", r"+2\vec{\mathbf{v} }", r"+ \vec{\mathbf{w} }")
        z_eq_1[0].set_color(YELLOW)
        z_eq_1[2].set_color(BLUE)
        z_eq_1[3].set_color(GREEN)
        z_eq_1[4].set_color(RED)
        z_eq_1.add_background_rectangle()
        
        
        z_eq_1.next_to(span_set, DOWN)
        
        self.add(z_eq_1[0], z_eq_1[2])
        
        self.play(
            Write(z_eq_1[2]),
            TransformFromCopy(z_label, z_eq_1[1]),
            TransformFromCopy(t_u_label, z_eq_1[3]),
            TransformFromCopy(t_v_label, z_eq_1[4]),
            TransformFromCopy(t_w_label, z_eq_1[5])
            
        )
        self.wait()
        
        
        transform_u_2 = self.get_vector(2 * array_u, color=BLUE)
        t_u_label_2 = MathTex(r"2\vec{\mathbf{u} }", color=BLUE)
        t_u_label_2.next_to(transform_u_2, DOWN*0.3 + RIGHT*0.3)
        
        original_u = vector_u.copy()
        original_u_label = u_label.copy()
        
        self.play(
            Transform(vector_u, transform_u_2),
            Transform(u_label, t_u_label_2),
            vector_v.animate.shift([2, 0, 0]),
            v_label.animate.shift([2, 0, 0] + 0.6*RIGHT + DOWN)
        )
        self.wait()
        
        
        w_eq = MathTex(r"\vec{\mathbf{w} }", r"=", r"2 \vec{\mathbf{u} }", r"+ \vec{\mathbf{v} }")
        w_eq[0].set_color(RED)
        w_eq[2].set_color(BLUE)
        w_eq[3].set_color(GREEN)
        w_eq.add_background_rectangle()
        w_eq.next_to(z_eq_1, DOWN)
        
        self.add(w_eq[0], w_eq[2])
        
        self.play(
            Write(w_eq[2]),
            TransformFromCopy(w_label, w_eq[1]),
            TransformFromCopy(t_u_label_2, w_eq[3]),
            TransformFromCopy(v_label, w_eq[4])
        )
        
        self.wait()
        
        
        z_eq_2 = MathTex(r"\vec{\mathbf{z} }", r"=", r"-4 \vec{\mathbf{u} }", r"+2 \vec{\mathbf{v} }", r"+ 2 \vec{\mathbf{u} }", r"+ \vec{\mathbf{v} }")
        z_eq_2[0].set_color(YELLOW)
        z_eq_2[2].set_color(BLUE)
        z_eq_2[3].set_color(GREEN)
        z_eq_2[4].set_color(BLUE)
        z_eq_2[5].set_color(GREEN)
        z_eq_2.next_to(span_set, DOWN)
        z_eq_2.add_background_rectangle()
        
        self.play(
            FadeOut(transform_w),
            FadeOut(t_w_label),
            transform_u_2.animate.shift([-4, 2, 0]),
            t_u_label_2.animate.shift([-4, 2, 0]),
            vector_v.animate.shift([-4, 2, 0]),
            v_label.animate.shift([-4, 2, 0]),
            FadeOut(w_eq),
            #FadeOut(z_eq_1),
            #FadeIn(z_eq_2)
            Transform(z_eq_1[0], z_eq_2[0]),
            Transform(z_eq_1[1:5], z_eq_2[1:5]),
            Transform(z_eq_1[5], z_eq_2[5:])
        )
        
        self.wait()
        
        
        z_eq_3 = MathTex(r"\vec{\mathbf{z} }", r"\in " r"span\{", r"\vec{\mathbf{u} }", r", \vec{\mathbf{v} }", r"\}")
        z_eq_3[0].set_color(YELLOW)
        z_eq_3[2].set_color(BLUE)
        z_eq_3[3].set_color(GREEN)
        z_eq_3.next_to(span_set, DOWN)
        z_eq_3.add_background_rectangle()
        
        self.play(
            FadeOut(transform_u),
            FadeOut(t_u_label),
            FadeOut(transform_v),
            FadeOut(t_v_label),
            FadeOut(transform_u_2),
            FadeOut(t_u_label_2),
            vector_v.animate.shift([2, -2, 0]),
            v_label.animate.shift([2, -2, 0] + 0.6*LEFT + UP),
            Transform(vector_u, original_u),
            Transform(u_label, original_u_label),
            FadeOut(z_eq_1),
            FadeIn(z_eq_3)
        )
        self.wait()
        
        
        z_eq_4 = MathTex(r"\mathbb{R}^2", r" \subseteq", r" span\{", r"\vec{\mathbf{u} }", r", \vec{\mathbf{v} }", r"\}")
        z_eq_4[3].set_color(BLUE)
        z_eq_4[4].set_color(GREEN)
        z_eq_4.next_to(span_set, DOWN)
        z_eq_4.add_background_rectangle()
        
        self.play(
            Transform(z_eq_3[0], z_eq_4[0]),
            Transform(z_eq_3[1:], z_eq_4[1:]),
            FadeOut(vector_z),
            FadeOut(z_label)
        )
        self.wait()
        
        
        new_span_set = Tex(r"$\{$", r"$\vec{\mathbf{u} },$",  r"$\vec{\mathbf{v} }$", r"$\}$ is a ", r"\textit{spanning set}", " for $\mathbb{R}^2$")
        new_span_set[1].set_color(BLUE)
        new_span_set[2].set_color(GREEN)
        new_span_set[4].set_color(YELLOW)
        new_span_set.to_edge(UP).to_edge(RIGHT).shift(UP*0.5)
        
        new_span_set.add_background_rectangle()
        self.bring_to_front(new_span_set)
        
        self.play(
            FadeOut(vector_w),
            FadeOut(w_label),
            FadeOut(span_set),
            FadeOut(z_eq_4),
            FadeIn(new_span_set)
        )
        
        basis_set = Tex(r"$\{$", r"$\vec{\mathbf{u} },$",  r"$\vec{\mathbf{v} }$", r"$\}$ is a ", r"\textit{basis}", " for $\mathbb{R}^2$")
        basis_set[1].set_color(BLUE)
        basis_set[2].set_color(GREEN)
        basis_set[4].set_color(YELLOW)
        basis_set.to_edge(UP).to_edge(RIGHT).shift(UP*0.5)
        
        basis_set.add_background_rectangle()
        self.bring_to_front(basis_set)
        
        self.play(
            FadeOut(new_span_set),
            FadeIn(basis_set)
        )
        
        
        self.wait()
        

class BasisAsSubsetAlgebra(Scene):
    def construct(self):
        line_1 = MathTex(r"\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}", r"\text{ is a spanning set for }", r"span\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}")
        line_1[0].set_color(BLUE)
        line_1[2].set_color(BLUE)
        
        self.play(Write(line_1))
        self.wait()
        
        line_2 = Tex(r"If linearly independent, got a basis")
        
        self.play(line_1.animate.shift(UP))
        self.wait()
        self.play(Write(line_2))
        self.wait()
        
        line_3 = Tex(r"Otherwise, can express at least one vector", r" in terms of others")
        line_3[1].set_color(YELLOW)
        
        self.play(line_1.animate.shift(UP), line_2.animate.shift(UP))
        self.wait()
        self.play(Write(line_3))
        self.wait()
        
        line_4 = Tex(r"So, throwing it away ", r"preserves span").shift(DOWN)
        line_4[1].set_color(YELLOW)
        
        self.play(Write(line_4))
        self.wait()
        
        line_5 = Tex(r"Keep going until basis found").shift(DOWN * 2)
        
        self.play(Write(line_5))
        self.wait()


class GeneralStrategy1(Scene):
    def construct(self):
        line_1 = Tex(r"Given $n$ vectors ", r"$\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}$", r" in $\mathbb{R}^d$").to_edge(UP)
        line_1[1].set_color(BLUE)
        
        line_2 = Tex(r"Get a ", r"basis", r" for ", r"$span\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}$").next_to(line_1, DOWN).shift(DOWN*0.5)
        line_2[1].set_color(YELLOW)
        line_2[3].set_color(BLUE)
        
        line_3 = Tex(r"Convert to ", r"an orthonormal basis").next_to(line_2, DOWN).shift(DOWN*0.5)
        line_3[1].set_color(YELLOW)
        
        line_4 = Tex(r"Extend to ", r"an orthonormal basis of $\mathbb{R}^d$").next_to(line_3, DOWN).shift(DOWN*0.5)
        line_4[1].set_color(YELLOW)
        
        line_5 = Tex(r"Use change of basis matrix from this to ", r"standard basis").next_to(line_4, DOWN).shift(DOWN*0.5)
        line_5[1].set_color(YELLOW)
        
        self.add(line_1)
        self.add(line_2)
        self.add(line_3)
        self.add(line_4)
        self.add(line_5)
        
        box = SurroundingRectangle(line_3)
        self.play(Create(box))
        self.wait()
       


class Orthonormalisation(VectorScene):
    def construct(self):
        self.plane = self.add_plane()
        
        array_v = np.array([2, 2, 0])
        array_w = np.array([-1, 3, 0])
        
        vector_v = self.add_vector(array_v, color=GREEN)

        vector_w = self.add_vector(array_w, color=RED)
        
        norm_v = self.get_vector(normalize(array_v))
        norm_v.set_color(GREEN)
        
        norm_text = Tex(r"Normalize")
        norm_text.to_edge(UP).to_edge(LEFT)
        
        self.play(Write(norm_text))
        self.wait()
        
        self.play(Transform(vector_v, norm_v))
        
        self.play(FadeOut(norm_text))
        self.wait()
        
        perp_text = Tex(r"Remove parallel components")
        perp_text.to_edge(UP).to_edge(LEFT)
        
        self.play(Write(perp_text))
        self.wait()
        
        v_line = Line(10*LEFT, 10*RIGHT)
        v_line.rotate(norm_v.get_angle())
        self.play(Create(v_line))
        self.wait()
        
        array_w_onto_v = np.dot(array_w, normalize(array_v)) * normalize(array_v)
        
        vector_proj = self.get_vector(array_w_onto_v)
        vector_proj.set_color(RED)
        
        proj_line = Line(vector_w.get_end(), vector_proj.get_end(), color=GRAY)
        
        self.play(Create(proj_line))
        
        self.play(TransformFromCopy(vector_w, vector_proj))
        self.wait()
        
        self.play(vector_proj.animate.scale(-1))
        
        self.play(vector_proj.animate.shift(vector_w.get_end() - vector_proj.get_start()))
        self.wait()
        
        orth_w = self.get_vector(array_w - array_w_onto_v)
        orth_w.set_color(RED)
        
        self.play(Transform(vector_w, orth_w), FadeOut(vector_proj), FadeOut(proj_line), FadeOut(v_line))
        self.wait()
        
        self.play(FadeOut(perp_text))
        
        norm_text = Tex(r"Normalize")
        norm_text.to_edge(UP).to_edge(LEFT)
        
        self.play(Write(norm_text))
        self.wait()
        
        norm_w = self.get_vector(normalize(array_w - array_w_onto_v))
        norm_w.set_color(RED)
        
        self.play(Transform(vector_w, norm_w))
        
        self.wait()



class OrthonormalisationAlgebra(Scene):
    def construct(self):
        title_text = Tex(r"Orthonormalisation").scale(2)
        title_text.to_edge(UP)
        self.play(Write(title_text))
        self.wait()
        
        start = MathTex(r"\text{Have our span-basis }\{", r"\vec{\mathbf{u} }_1", r", ", r"\vec{\mathbf{u} }_2", r", \dots, ", r"\vec{\mathbf{u} }_k", r"\}")
        start.set_color_by_tex(r"\vec{\mathbf", BLUE)
        start.next_to(title_text, DOWN)
        self.play(Write(start))
        self.wait()
        
        initially = Tex(r"Initially, we normalize ", r"$\vec{\mathbf{u} }_1$")
        initially.set_color_by_tex(r"\vec{\mathbf", BLUE)
        initially.next_to(start, DOWN)
        self.play(Write(initially))
        self.wait()
        
        to_normalize = MathTex(r"\vec{\mathbf{u} }_1", r" \longrightarrow", r" \frac{1}{\abs{\vec{\mathbf{u} }_1} } \vec{\mathbf{u_1} }")
        to_normalize.set_color_by_tex(r"\vec{\mathbf", BLUE)
        to_normalize.next_to(initially, DOWN)
        self.play(Write(to_normalize))
        self.wait()
        
        self.play(FadeOut(to_normalize))
        
        algorithm_desc = Tex(r"Then, repeat the following steps, starting from $r = 2$:")
        algorithm_desc.next_to(initially, DOWN)
        self.play(Write(algorithm_desc))
        self.wait()
        
        step_1 = Tex(r"1: Remove the components of ", r"$\vec{\mathbf{u} }_r$", r" parallel to previous vectors", color=YELLOW)
        step_1.set_color_by_tex(r"\vec{\mathbf", BLUE)
        step_1.next_to(algorithm_desc, DOWN)
        step_1.to_edge(LEFT)
        self.play(Write(step_1))
        self.wait()
        
        math_step_1 = MathTex(r"\vec{\mathbf{u} }_r", r" \longrightarrow", r" \vec{\mathbf{u} }_r", r" - (", r"\vec{\mathbf{u} }_1", r" \cdot ", r"\vec{\mathbf{u} }_r", r") ", r"\vec{\mathbf{u} }_1", r" - \dots - (", r"\vec{\mathbf{u} }_{r-1}", r" \cdot ", r"\vec{\mathbf{u} }_r", r") ", r"\vec{\mathbf{u} }_{r-1}")
        math_step_1.set_color_by_tex(r"\vec{\mathbf", BLUE)
        math_step_1.next_to(step_1, DOWN)
        self.play(Write(math_step_1))
        self.wait()
        
        step_2 = Tex(r"2: Normalize ", r"$\vec{\mathbf{u} }_r$", color=YELLOW)
        step_2.set_color_by_tex(r"\vec{\mathbf", BLUE)
        step_2.next_to(math_step_1, DOWN)
        step_2.to_edge(LEFT)
        self.play(Write(step_2))
        self.wait()
        
        step_3 = Tex(r"3: Keep going until set is exhausted", color=YELLOW)
        step_3.next_to(step_2, DOWN)
        step_3.to_edge(LEFT)
        self.play(Write(step_3))
        self.wait()


class GeneralStrategy2(Scene):
    def construct(self):
        line_1 = Tex(r"Given $n$ vectors ", r"$\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}$", r" in $\mathbb{R}^d$").to_edge(UP)
        line_1[1].set_color(BLUE)
        
        line_2 = Tex(r"Get a ", r"basis", r" for ", r"$span\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}$").next_to(line_1, DOWN).shift(DOWN*0.5)
        line_2[1].set_color(YELLOW)
        line_2[3].set_color(BLUE)
        
        line_3 = Tex(r"Convert to ", r"an orthonormal basis").next_to(line_2, DOWN).shift(DOWN*0.5)
        line_3[1].set_color(YELLOW)
        
        line_4 = Tex(r"Extend to ", r"an orthonormal basis of $\mathbb{R}^d$").next_to(line_3, DOWN).shift(DOWN*0.5)
        line_4[1].set_color(YELLOW)
        
        line_5 = Tex(r"Use change of basis matrix from this to ", r"standard basis").next_to(line_4, DOWN).shift(DOWN*0.5)
        line_5[1].set_color(YELLOW)
        
        self.add(line_1)
        self.add(line_2)
        self.add(line_3)
        self.add(line_4)
        self.add(line_5)
        
        box = SurroundingRectangle(line_4)
        self.play(Create(box))
        self.wait()


class BasisExtension(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(z_min=-4.5, z_max=4.5).add_coordinates()
        
        vector_v = [1, -2, 1]
        vector_w = [2, 0, 2]
        i_hat = [1, 0, 0]
        j_hat = [0, 1, 0]
        k_hat = [0, 0, 1]
        
        arrow_v = Arrow3D(np.array([0, 0, 0]), np.array([1, -2, 1]), color=GREEN)
        arrow_w = Arrow3D(np.array([0, 0, 0]), np.array([2, 0, 2]), buff=0, color=RED)
        
        new_array_v = np.array([-0.5, 1, -0.5]) # -0.5 times old array
        new_array_w = np.array([0.5, 0, 0.5]) # 0.25 times old array
        
        new_arrow_v = Arrow3D(np.array([0, 0, 0]), np.array(new_array_v), color=GREEN)
        new_arrow_w = Arrow3D(np.array(new_array_v), np.array(new_array_v + new_array_w), buff=0, color=RED)
        
        vector_i = Arrow3D(np.array([0, 0, 0]), np.array(i_hat), color=BLUE)
        vector_j = Arrow3D(np.array([0, 0, 0]), np.array(j_hat), color=YELLOW)
        vector_k = Arrow3D(np.array([0, 0, 0]), np.array(k_hat), color=PURPLE)
        
        v_w_plane = Polygon([-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0], color="white")
        v_w_plane.set_fill(WHITE, opacity=0.5)
        
        mobjects = [axes, arrow_v, arrow_w, v_w_plane, vector_i, vector_j, vector_k, new_arrow_v, new_arrow_w]
        for mobj in mobjects:
            mobj = mobj.scale(0.7, about_point=[0,0,0])
        
        self.add(arrow_v)
        self.add(arrow_w)
        self.add(axes)
        
        self.set_camera_orientation(phi=60 * DEGREES, theta=-100 * DEGREES)
        
        
        mid_axis = normalize([0, 0, 1] + get_unit_normal(vector_v, vector_w))
        
        self.play(Create(v_w_plane))
        
        self.play(Rotate(v_w_plane, axis=mid_axis, about_point=[0,0,0]))
        
        self.wait()
        
        basis_text = Tex(r"Add standard basis")
        basis_text.to_edge(UP).to_edge(RIGHT)
        
        self.add_fixed_in_frame_mobjects(basis_text)
        self.play(Write(basis_text))
        
        self.add(vector_i, vector_j, vector_k)
        self.wait()
        
        self.move_camera(phi=90*DEGREES, theta=-90*DEGREES)
        self.wait()
        
        span_text = Tex(r"Yellow ", r"in span, ", r"blue ", r"and ", r"purple ", r"aren't")
        span_text[0].set_color(YELLOW)
        span_text[2].set_color(BLUE)
        span_text[4].set_color(PURPLE)
        
        
        span_text.next_to(basis_text, DOWN).to_edge(RIGHT).add_background_rectangle()
        
        self.add_fixed_in_frame_mobjects(span_text)
        self.play(Write(span_text[1:]))
        
        self.move_camera(phi=60*DEGREES, theta=-180*DEGREES)
        
        self.wait()
        
        self.play(FadeOut(v_w_plane))
        self.wait()
        
        
        
        self.play(Transform(arrow_v, new_arrow_v), Transform(arrow_w, new_arrow_w))
        self.wait()
        
        add_text = Tex(r"Add either ", r"blue", r" or ", r"purple", r" to extend basis")
        add_text[1].set_color(BLUE)
        add_text[3].set_color(PURPLE)
        
        add_text.next_to(span_text, DOWN).to_edge(RIGHT).add_background_rectangle()
        
        self.add_fixed_in_frame_mobjects(add_text)
        self.play(Write(add_text[1:]))
        self.wait()
      

class BasisExtensionAlgebra(Scene):
    def construct(self):
        line_1 = Tex(r"Got basis ", r"$\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_k\}$", r" for the span subspace")
        line_1[1].set_color(BLUE)
        
        self.play(Write(line_1))
        self.wait()
        
        line_2 = Tex(r"See which of the standard basis is not in span")
        
        self.play(line_1.animate.shift(UP))
        self.wait()
        self.play(Write(line_2))
        self.wait()
        
        line_3 = Tex(r"If all in span, then full basis found")
        
        self.play(line_1.animate.shift(UP), line_2.animate.shift(UP))
        self.wait()
        self.play(Write(line_3))
        self.wait()
        
        line_4 = Tex(r"Otherwise, add one not in span, and repeat from line 2").shift(DOWN)
        
        self.play(Write(line_4))
        self.wait()

class GeneralStrategy3(Scene):
    def construct(self):
        line_1 = Tex(r"Given $n$ vectors ", r"$\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}$", r" in $\mathbb{R}^d$").to_edge(UP)
        line_1[1].set_color(BLUE)
        
        line_2 = Tex(r"Get a ", r"basis", r" for ", r"$span\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}$").next_to(line_1, DOWN).shift(DOWN*0.5)
        line_2[1].set_color(YELLOW)
        line_2[3].set_color(BLUE)
        
        line_3 = Tex(r"Convert to ", r"an orthonormal basis").next_to(line_2, DOWN).shift(DOWN*0.5)
        line_3[1].set_color(YELLOW)
        
        line_4 = Tex(r"Extend to ", r"an orthonormal basis of $\mathbb{R}^d$").next_to(line_3, DOWN).shift(DOWN*0.5)
        line_4[1].set_color(YELLOW)
        
        line_5 = Tex(r"Use change of basis matrix from this to ", r"standard basis").next_to(line_4, DOWN).shift(DOWN*0.5)
        line_5[1].set_color(YELLOW)
        
        self.add(line_1)
        self.add(line_2)
        self.add(line_3)
        self.add(line_4)
        self.add(line_5)
        
        box = SurroundingRectangle(line_5)
        self.play(Create(box))
        self.wait()


class WhyLastCoordsAreZero(Scene):
    def construct(self):
        line_1 = MathTex(r"\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}", r"\text{ is a spanning set for }", r"span\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}").shift(UP * 2)
        line_1[0].set_color(BLUE)
        line_1[2].set_color(BLUE)
        
        self.play(Write(line_1))
        self.wait()
        
        line_2 = Tex(r"Can find a basis as a ", r"subset", r" of ", r"$\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}$").next_to(line_1, DOWN)
        line_2[1].set_color(YELLOW)
        line_2[3].set_color(BLUE)
        
        self.play(Write(line_2))
        self.wait()
        
        line_3 = Tex(r"So, ", r"dimension", r" of ", r"$span\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}$", r" is at most ", r"$n$").next_to(line_2, DOWN)
        line_3[1].set_color(YELLOW)
        line_3[3].set_color(BLUE)
        line_3[5].set_color(YELLOW)
        
        self.play(Write(line_3))
        self.wait()
        
        line_4 = Tex(r"So, after ", r"change of basis", r", only first ", r"$n$", r" standard basis vectors used to describe them").scale(0.8).next_to(line_3, DOWN)
        line_4[1].set_color(PURPLE)
        line_4[3].set_color(YELLOW)
        
        self.play(Write(line_4))
        self.wait()


class GeneralStrategy4(Scene):
    def construct(self):
        line_1 = Tex(r"Given $n$ vectors ", r"$\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}$", r" in $\mathbb{R}^d$").to_edge(UP)
        line_1[1].set_color(BLUE)
        
        line_2 = Tex(r"Get a ", r"basis", r" for ", r"$span\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}$").next_to(line_1, DOWN).shift(DOWN*0.5)
        line_2[1].set_color(YELLOW)
        line_2[3].set_color(BLUE)
        
        line_3 = Tex(r"Convert to ", r"an orthonormal basis").next_to(line_2, DOWN).shift(DOWN*0.5)
        line_3[1].set_color(YELLOW)
        
        line_4 = Tex(r"Extend to ", r"an orthonormal basis of $\mathbb{R}^d$").next_to(line_3, DOWN).shift(DOWN*0.5)
        line_4[1].set_color(YELLOW)
        
        line_5 = Tex(r"Use change of basis matrix from this to ", r"standard basis").next_to(line_4, DOWN).shift(DOWN*0.5)
        line_5[1].set_color(YELLOW)
        
        code_text = Tex(r"What might this look like in code?", color=YELLOW).scale(1.5).next_to(line_5, DOWN).shift(DOWN*0.5)
        
        self.add(line_1)
        self.add(line_2)
        self.add(line_3)
        self.add(line_4)
        self.add(line_5)
        
        self.wait()
        
        self.play(Write(code_text))
        self.wait()


class ClunkyAlgorithm(Scene):
    def construct(self):
        clunky_algorithm = Code(file_name="gram_code.py", style="monokai", scale_factor=0.4)
        
        self.play(Create(clunky_algorithm[0]))
        self.wait()
        for line_number, code_string in zip(clunky_algorithm[1], clunky_algorithm[2]):
            self.play(Write(line_number), Write(code_string))
            self.wait()
        self.wait()
        
        self.play(clunky_algorithm.animate.scale(0.7))
        self.wait()
        
        terrible_text = Tex(r"All kinds of terrible!", color=RED).scale(1.5).next_to(clunky_algorithm, UP)
        
        self.play(Write(terrible_text))
        self.wait()


class EndOfDerivation(Scene):
    def construct(self):
    
        
        ##########
        # Steps of algorithm so far
        ##########
        
        ##########
        # Starting
        ##########
        
        start_text = Tex(r"Started with $d$-dimensional vectors ", r"$\{\vec{\mathbf{u_1} }, \vec{\mathbf{u_2} }, \dots, \vec{\mathbf{u_n} }\}$").scale(0.7).to_edge(UP).to_edge(LEFT)
        start_text[1].set_color(BLUE)
        self.play(Write(start_text))
        self.wait()
        
        array_v = [1, -2, 1]
        array_w = [2, 0, 2]
        
        mid_axis = normalize([0, 0, 1] + get_unit_normal(array_v, array_w))
        
        matrix_1 = DecimalMatrix(rotation_matrix(180 * DEGREES, mid_axis), element_to_mobject_config={'num_decimal_places': 2}).scale(0.8)
        
        v_1 = DecimalMatrix([[elem] for elem in array_v], element_to_mobject_config={'num_decimal_places': 2}).scale(0.8).set_color(GREEN).to_edge(LEFT).shift(DOWN*0.3)
        
        w_1 = DecimalMatrix([[elem] for elem in array_w], element_to_mobject_config={'num_decimal_places': 2}).scale(0.8).set_color(RED).next_to(v_1)
        
        self.play(Create(v_1), Create(w_1))
        self.wait()
        
        ############
        # Applying Orthogonal Matrix
        ############
        
        step_1 = Tex(r"Applied Orthogonal Matrix,  getting $d$-dimensional vectors ", r"$\{\vec{\mathbf{v_1} }, \vec{\mathbf{v_2} }, \dots, \vec{\mathbf{v_n} }\}$").scale(0.7).next_to(start_text, DOWN).to_edge(LEFT)
        self.play(Write(step_1))
        self.wait()
        
        transform_arrow_1 = Arrow((w_1.get_corner(UR) + w_1.get_corner(DR))/2, (w_1.get_corner(UR) + w_1.get_corner(DR))/2 + 1.5*RIGHT)
        
        self.play(Create(transform_arrow_1))
        self.wait()
        
        v_2 = DecimalMatrix([[elem] for elem in np.matmul(rotation_matrix(180 * DEGREES, mid_axis), array_v)], element_to_mobject_config={'num_decimal_places': 2}).scale(0.8).next_to(transform_arrow_1, RIGHT).set_color(GREEN)
        
        w_2 = DecimalMatrix([[elem] for elem in np.matmul(rotation_matrix(180 * DEGREES, mid_axis), array_w)], element_to_mobject_config={'num_decimal_places': 2}).scale(0.8).next_to(v_2, RIGHT).set_color(RED)
        
        self.play(Create(v_2), Create(w_2))
        self.wait()
        
        transform_brace_1 = Brace(transform_arrow_1, DOWN)
        
        matrix_1.next_to(transform_brace_1, DOWN).shift(DOWN*0.15)
        
        self.play(Create(transform_brace_1), Create(matrix_1))
        self.wait()
        
        self.play(FadeOut(transform_brace_1), FadeOut(matrix_1))
        
        ##########
        # Removing trailing zeroes
        ##########
        
        step_2 = Tex(r"Dropped trailing zeroes to make $n$-dimensional vectors ", r"$\{\vec{\mathbf{w_1} }, \vec{\mathbf{w_2} }, \dots, \vec{\mathbf{w_n} }\}$").scale(0.7).next_to(step_1, DOWN).to_edge(LEFT)
        step_2[1].set_color(PURPLE)
        self.play(Write(step_2))
        self.wait()
        
        
        transform_arrow_2 = Arrow((w_2.get_corner(UR) + w_2.get_corner(DR))/2, (w_2.get_corner(UR) + w_2.get_corner(DR))/2 + 1.5*RIGHT)
        
        self.play(Create(transform_arrow_2))
        self.wait()
        
        v_3 = DecimalMatrix([[elem] for elem in np.matmul(rotation_matrix(180 * DEGREES, mid_axis), array_v)][:2], element_to_mobject_config={'num_decimal_places': 2}).scale(0.8).next_to(transform_arrow_2, RIGHT).set_color(GREEN)
        
        w_3 = DecimalMatrix([[elem] for elem in np.matmul(rotation_matrix(180 * DEGREES, mid_axis), array_w)][:2], element_to_mobject_config={'num_decimal_places': 2}).scale(0.8).next_to(v_3, RIGHT).set_color(RED)
        
        self.play(Create(v_3), Create(w_3))
        self.wait()
        
        #########
        # Making the square matrix
        #########
        
        step_3 = Tex(r"Formed \textit{square} matrix ", r"$\mathbf{W}$", r" with columns ", r"$\{\vec{\mathbf{w_1} }, \vec{\mathbf{w_2} }, \dots, \vec{\mathbf{w_n} }\}$").scale(0.7).next_to(step_2, DOWN).to_edge(LEFT)
        step_3[1].set_color(PURPLE)
        step_3[3].set_color(PURPLE)
        self.play(Write(step_3))
        self.wait()
        
        transform_arrow_3 = Arrow(v_1.get_corner(DL) + DOWN, v_1.get_corner(DL) + DOWN + 1.5*RIGHT)
        
        self.play(Create(transform_arrow_3))
        self.wait()
        
        target_matrix = np.matmul(rotation_matrix(180 * DEGREES, mid_axis), np.transpose([array_v, array_w]))[:2]
        
        target_matrix = DecimalMatrix(target_matrix, element_to_mobject_config={'num_decimal_places': 2}).scale(0.8).next_to(transform_arrow_3, RIGHT)
        
        target_matrix.set_column_colors(GREEN, RED)
        
        self.play(Create(target_matrix))
        self.wait()
        
        ##########
        # Taking determinant
        ##########
        
        step_4 = Tex(r"Took determinant of ", r"$\mathbf{W}$").scale(0.7).next_to(step_3, DOWN).to_edge(LEFT)
        step_4[1].set_color(PURPLE)
        
        self.play(Write(step_4))
        self.wait()
        
        transform_arrow_4 = Arrow((target_matrix.get_corner(UR) + target_matrix.get_corner(DR))/2, (target_matrix.get_corner(UR) + target_matrix.get_corner(DR))/2 + 1.5*RIGHT)
        
        self.play(Create(transform_arrow_4))
        self.wait()
        
        target_matrix_cpy = target_matrix.copy()
        
        det_text = get_det_text(target_matrix_cpy, determinant="5.66", initial_scale_factor=1)
        
        det_grp = VGroup(target_matrix_cpy, det_text)
        
        det_grp.next_to(transform_arrow_4, RIGHT)
        
        self.play(Create(det_grp))
        self.wait()
        
        ##########
        # All done with that!
        #########
        
        step_group = VGroup(start_text, step_1, step_2, step_3, step_4)
        
        step_group.save_state()
        
        # Get rid of the example stuff below
        
        example_grp_1 = VGroup(
            v_1,
            w_1,
            transform_arrow_1,
            v_2,
            w_2,
            transform_arrow_2,
            v_3,
            w_3,
            transform_arrow_3,
            target_matrix,
            transform_arrow_4,
            det_grp
        )
        
        self.play(FadeOut(example_grp_1))
        self.wait()
        
        ##########
        # Show preservation of dot product
        ##########
        
        arrow_1 = Arrow(start_text.get_corner(DL), step_1.get_corner(UL) + DOWN)
        
        dot_product_preserved = Tex(r"Dot product preserved", color=YELLOW).move_to((start_text.get_corner(UL) + step_1.get_corner(DL) + DOWN)/2).to_edge(LEFT)
        
        dot_product_preserved.next_to(arrow_1, RIGHT)
        
        self.play(step_1.animate.shift(DOWN), step_2.animate.shift(DOWN), step_3.animate.shift(DOWN), step_4.animate.shift(DOWN), Write(dot_product_preserved), Create(arrow_1))
        
        self.wait()
        
        because_text = Tex(r"because orthogonal matrix").next_to(dot_product_preserved, RIGHT)
        self.play(Write(because_text))
        
        arrow_2 = Arrow(step_1.get_corner(DL), step_2.get_corner(UL) + DOWN)
        
        dot_product_preserved_2 = dot_product_preserved.copy().move_to((step_1.get_corner(UL) + step_2.get_corner(DL) + DOWN)/2).to_edge(LEFT)
        
        dot_product_preserved_2.next_to(arrow_2, RIGHT)
        
        self.play(step_2.animate.shift(DOWN), step_3.animate.shift(DOWN), step_4.animate.shift(DOWN), Write(dot_product_preserved_2), Create(arrow_2))
        self.wait()
        
        example_text = MathTex(r"\begin{bmatrix} 1 \\ 4 \\ 2 \\ 0 \\ 0 \end{bmatrix} \cdot \begin{bmatrix} 8 \\ 5 \\ 7 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ 4 \\ 2 \end{bmatrix} \cdot \begin{bmatrix} 8 \\ 5 \\ 7 \end{bmatrix}", color=BLUE).scale(0.9).to_edge(DOWN).to_edge(RIGHT)
        
        self.play(Write(example_text))
        self.wait()
        
        self.play(FadeOut(example_text), FadeOut(because_text), FadeOut(arrow_1), FadeOut(arrow_2), FadeOut(dot_product_preserved), FadeOut(dot_product_preserved_2), Restore(step_group))
        self.wait()
        
        dot_product_identity = MathTex(r"\vec{\mathbf{u} }_i \cdot \vec{\mathbf{u} }_j", r"=", r"\vec{\mathbf{w} }_i \cdot \vec{\mathbf{w} }_j").to_edge(UP).to_edge(RIGHT)
        dot_product_identity[0].set_color(BLUE)
        dot_product_identity[2].set_color(PURPLE)
        
        self.play(Write(dot_product_identity))
        self.wait()
        
        
        ###############
        # Time for transpose
        ###############
        
        self.play(FadeOut(step_1), FadeOut(step_2), FadeOut(step_4))
        self.wait()
        
        self.play(start_text.animate.scale(1/0.7).to_edge(LEFT).shift(DOWN), dot_product_identity.animate.scale(1/0.7).to_edge(LEFT))
        self.play(step_3.animate.scale(1/0.7).next_to(start_text, DOWN).to_edge(LEFT))
        self.wait()
        
        consider_1 = Tex(r"Consider matrix ", r"$\mathbf{U}$", r" with columns ", r"$\vec{\mathbf{u} }_i$").next_to(step_3, DOWN).to_edge(LEFT)
        consider_1[1].set_color(BLUE)
        consider_1[3].set_color(BLUE)
        
        self.play(Write(consider_1))
        self.wait()
        
        matrix_u = DecimalMatrix(list(zip(array_v, array_w)), element_to_mobject_config={'num_decimal_places': 2})
        matrix_u_entries = matrix_u.get_entries()
        matrix_u_entries.set_color(BLUE)


class EndOfDerivation2(Scene):
    def construct(self):
        
        ###########
        # Matching end of previous scene
        ##########
        
        dot_product_identity = MathTex(r"\vec{\mathbf{u} }_i \cdot \vec{\mathbf{u} }_j", r"=", r"\vec{\mathbf{w} }_i \cdot \vec{\mathbf{w} }_j").scale(1/0.7).to_edge(UP).to_edge(LEFT)
        dot_product_identity[0].set_color(BLUE)
        dot_product_identity[2].set_color(PURPLE)
        self.add(dot_product_identity)
    
        start_text = Tex(r"Started with $d$-dimensional vectors ", r"$\{\vec{\mathbf{u_1} }, \vec{\mathbf{u_2} }, \dots, \vec{\mathbf{u_n} }\}$").next_to(dot_product_identity, DOWN).to_edge(LEFT)
        start_text[1].set_color(BLUE)
        self.add(dot_product_identity)
        self.add(start_text)
        
        array_v = [1, -2, 1]
        array_w = [2, 0, 2]
        
        mid_axis = normalize([0, 0, 1] + get_unit_normal(array_v, array_w))
        
        step_3 = Tex(r"Formed \textit{square} matrix ", r"$\mathbf{W}$", r" with columns ", r"$\{\vec{\mathbf{w_1} }, \vec{\mathbf{w_2} }, \dots, \vec{\mathbf{w_n} }\}$").next_to(start_text, DOWN).to_edge(LEFT)
        step_3[1].set_color(PURPLE)
        step_3[3].set_color(PURPLE)
        self.add(step_3)
        
        consider_text = Tex(r"Consider matrix ", r"$\mathbf{U}$", r" with columns ", r"$\vec{\mathbf{u} }_i$").next_to(step_3, DOWN).to_edge(LEFT)
        consider_text[1].set_color(BLUE)
        consider_text[3].set_color(BLUE)
        
        self.add(consider_text)
        self.wait()
        
        
        u_eq = MathTex(r"\mathbf{U} =", color=BLUE).to_edge(LEFT)
        
        matrix_u = DecimalMatrix(list(zip(array_v, array_w)), element_to_mobject_config={'num_decimal_places': 2}).scale(0.7).next_to(u_eq, RIGHT)
        matrix_u_entries = matrix_u.get_entries()
        matrix_u_entries.set_color(BLUE)
        
        u_grp_1 = VGroup(u_eq, matrix_u)
        u_grp_1.next_to(consider_text, DOWN).to_edge(LEFT)
        
        w_eq = MathTex(r"\mathbf{W} =", color=PURPLE).next_to(matrix_u, RIGHT)
        
        target_matrix = np.matmul(rotation_matrix(180 * DEGREES, mid_axis), np.transpose([array_v, array_w]))[:2]
        
        matrix_w = DecimalMatrix(target_matrix, element_to_mobject_config={'num_decimal_places': 2}).scale(0.8).next_to(w_eq, RIGHT)
        matrix_w.get_entries().set_color(PURPLE)
        
        self.play(Write(u_eq), Create(matrix_u))
        self.wait()
        
        self.play(Write(w_eq), Create(matrix_w))
        self.wait()
        
        self.play(FadeOut(u_eq), FadeOut(matrix_u), FadeOut(w_eq), FadeOut(matrix_w))
        
        #######
        # Big equation time
        #######
        
        big_eq_start = MathTex(r"\mathbf{W}^T \mathbf{W} =", color=PURPLE)
        
        matrix_w_t = DecimalMatrix(np.transpose(target_matrix), element_to_mobject_config={'num_decimal_places': 2}).scale(0.8).next_to(big_eq_start, RIGHT)
        matrix_w_t.get_entries().set_color(PURPLE)
        
        matrix_w_cpy = matrix_w.copy()
        matrix_w_cpy.next_to(matrix_w_t, RIGHT)
        
        equals_1 = MathTex(r"=").next_to(matrix_w_cpy, RIGHT)
        
        gram_matrix = DecimalMatrix(np.matmul(np.transpose(target_matrix), target_matrix), element_to_mobject_config={'num_decimal_places': 2}).scale(0.8).next_to(equals_1, RIGHT)
        
        equals_2 = MathTex(r"=").next_to(gram_matrix, RIGHT)
        
        matrix_u_t = DecimalMatrix(np.transpose(list(zip(array_v, array_w))), element_to_mobject_config={'num_decimal_places': 2}).scale(0.8).next_to(equals_2, RIGHT)
        
        matrix_u_t.get_entries().set_color(BLUE)
        
        matrix_u_cpy = matrix_u.copy().next_to(matrix_u_t, RIGHT)
        
        big_eq_end = MathTex(r"= ", r"\mathbf{U}^T \mathbf{U}", color=BLUE).next_to(matrix_u_cpy, RIGHT)
        
        big_eq_grp_1 = VGroup(
            big_eq_start,
            matrix_w_t,
            matrix_w_cpy,
            equals_1,
            gram_matrix,
        )
        
        big_eq_grp_2 = VGroup(
            equals_2,
            matrix_u_t,
            matrix_u_cpy,
            big_eq_end
        )
        
        big_eq_grp = VGroup(big_eq_grp_1, big_eq_grp_2).arrange(DOWN).next_to(consider_text, DOWN).to_edge(LEFT)
        
        
        self.play(Create(big_eq_grp_1))
        self.wait()
        self.play(Create(big_eq_grp_2))
        self.wait()
        
        ##########
        # Explaining why it works
        ##########
        
        box_top_left = SurroundingRectangle(gram_matrix[0][0])
        self.play(Create(box_top_left))
        self.wait()
        
        box_w_t_row_1 = SurroundingRectangle(matrix_w_t[0][0:2])
        box_w_col_1 = SurroundingRectangle(matrix_w_cpy[0][0:3:2])
        
        box_u_t_row_1 = SurroundingRectangle(matrix_u_t[0][0:3])
        box_u_col_1 = SurroundingRectangle(matrix_u_cpy[0][0:5:2])
        
        self.play(TransformFromCopy(box_top_left, box_w_t_row_1), TransformFromCopy(box_top_left, box_w_col_1))
        self.wait()
        self.play(TransformFromCopy(box_top_left, box_u_t_row_1), TransformFromCopy(box_top_left, box_u_col_1))
        self.wait()
        
        box_ident = SurroundingRectangle(dot_product_identity)
        
        self.play(Create(box_ident))
        self.wait()
        
        self.play(
            FadeOut(box_top_left),
            FadeOut(box_w_t_row_1),
            FadeOut(box_w_col_1),
            FadeOut(box_u_t_row_1),
            FadeOut(box_u_col_1),
            FadeOut(box_ident)
        )
        
        fade_grp = VGroup(VGroup(*big_eq_grp_1[1:]), VGroup(*big_eq_grp_2[:-1]))
        
        to_fade = big_eq_end[0]
        big_eq_end = big_eq_end[1]
        
        self.play(FadeOut(fade_grp), FadeOut(to_fade), big_eq_end.animate.next_to(big_eq_start, RIGHT))
        self.wait()
        
        implies_text_1 = MathTex(r"\Rightarrow \operatorname{det} (", r"\mathbf{W}^T \mathbf{W}", r")", r" = \operatorname{det} (", r"\mathbf{U}^T \mathbf{U}", r")").next_to(big_eq_end, RIGHT)
        implies_text_1[1].set_color(PURPLE)
        implies_text_1[4].set_color(BLUE)
        
        self.play(Write(implies_text_1))
        self.wait()
        
        rectangular_brace = Brace(implies_text_1[3:], DOWN)
        rectangular_brace_text = rectangular_brace.get_tex(r"\text{Can't break up as }", r"\mathbf{U}", r"\text{ is rectangular}")
        rectangular_brace_text[1].set_color(BLUE)
        
        self.play(Create(rectangular_brace), Write(rectangular_brace_text))
        self.wait()
        
        self.play(FadeOut(rectangular_brace), FadeOut(rectangular_brace_text))
        
        square_brace = Brace(implies_text_1[:3], DOWN)
        square_brace_text = square_brace.get_tex(r"\text{Can break up as }", r"\mathbf{W}", r"\text{ is square}")
        square_brace_text.set_color(YELLOW)
        square_brace_text[1].set_color(PURPLE)
        
        self.play(Create(square_brace), Write(square_brace_text))
        self.wait()
        self.play(FadeOut(square_brace), FadeOut(square_brace_text))
        self.wait()
        
        implies_text_2 = MathTex(r"\Rightarrow \operatorname{det} (", r"\mathbf{W}^T", r") \operatorname{det} (", r" \mathbf{W}", r")", r" = \operatorname{det} (", r"\mathbf{U}^T \mathbf{U}", r")").next_to(implies_text_1, DOWN).align_to(implies_text_1, LEFT)
        implies_text_2[1].set_color(PURPLE)
        implies_text_2[3].set_color(PURPLE)
        implies_text_2[6].set_color(BLUE)
        self.play(Write(implies_text_2))
        self.wait()
        
        implies_text_3 = MathTex(r"\Rightarrow \operatorname{det} (", r"\mathbf{W}", r")^2 = \operatorname{det} (", r"\mathbf{U}^T \mathbf{U}", r")").next_to(implies_text_2, DOWN).align_to(implies_text_2, LEFT)
        implies_text_3[1].set_color(PURPLE)
        implies_text_3[3].set_color(BLUE)
        self.play(Write(implies_text_3))
        self.wait()
        
        implies_text_4 = MathTex(r"\Rightarrow", r" |\operatorname{det} (", r"\mathbf{W}", r")|", r" =", r" \sqrt{\det(\mathbf{U}^T \mathbf{U})}").next_to(implies_text_3, DOWN).align_to(implies_text_3, LEFT)
        implies_text_4[2].set_color(PURPLE)
        implies_text_4[5].set_color(BLUE)
        self.play(Write(implies_text_4))
        self.wait()
        
        ##########
        # MAIN RESULT
        #########
        
        to_fade_2 = implies_text_4[0]
        implies_text_4 = implies_text_4[1:]
        
        
        self.play(
            FadeOut(dot_product_identity),
            FadeOut(start_text),
            FadeOut(step_3),
            FadeOut(consider_text),
            FadeOut(big_eq_start),
            FadeOut(big_eq_end),
            FadeOut(implies_text_1),
            FadeOut(implies_text_2),
            FadeOut(implies_text_3),
            FadeOut(to_fade_2),
            implies_text_4.animate.scale(2).move_to([0, 0, 0])
        )
        self.wait()
        
        volume_brace = Brace(implies_text_4[:3], DOWN)
        volume_brace_text = volume_brace.get_text(r"Volume!")
        volume_brace_text.set_color(YELLOW)
        
        compute_brace = Brace(implies_text_4[4], DOWN)
        compute_brace_text = compute_brace.get_text(r"Easy to compute!")
        compute_brace_text.set_color(YELLOW)
        
        self.play(Create(volume_brace), Write(volume_brace_text))
        self.wait()
        self.play(Create(compute_brace), Write(compute_brace_text))
        self.wait()


    
class EasyAlgorithm(Scene):
    def construct(self):
        top_text = MathTex(r"\sqrt{\det(\mathbf{U}^T \mathbf{U})}", color=BLUE).scale(2).to_edge(UP)
        
        self.play(Write(top_text))
        self.wait()
        
        easy_algorithm = Code(file_name="easy_gram_code.py", style="monokai")
        
        self.play(Create(easy_algorithm))
        self.wait()
        
        line_text = Tex(r"Basically a one-liner!", color=YELLOW).scale(2).to_edge(DOWN)
        
        self.play(Write(line_text))
        self.wait()


class WhyDidThatWork(Scene):
    def construct(self):
        why_text = Tex(r"Why did that work?").scale(1.5).to_edge(UP)
        
        self.play(Write(why_text))
        self.wait()
        
        reduce_text = Tex(r"Reduced to known problem").scale(1.5).shift(UP)
        self.play(Write(reduce_text))
        self.wait()
        
        invar_text = Tex(r"Exploited an invariant").scale(1.5).set_color(YELLOW)
        self.play(Write(invar_text))
        self.wait()




class ExampleGramian(Scene):
    def construct(self):
        array_v = [1, -2, 1]
        array_w = [2, 0, 2]
        
        coord_text = MathTex(r"\vec{\mathbf{v} } = \begin{bmatrix} 1 \\ -2 \\ 1 \end{bmatrix} \space", r"\vec{\mathbf{w} } = \begin{bmatrix} 2 \\ 0 \\ 2 \end{bmatrix}").to_edge(UP)
        
        coord_text[0].set_color(GREEN)
        coord_text[1].set_color(RED)
        
        self.play(Write(coord_text))
        self.wait()
        
        gram_text = Tex(r"Gramian $=$", color=BLUE).scale(0.9).to_edge(LEFT)
        
        matrix = np.transpose([np.transpose(array_v), np.transpose(array_w)])
        
        gram_matrix = np.matmul(np.transpose(matrix), matrix)
        
        display_matrix_t = Matrix(np.transpose(matrix)).scale(0.9).next_to(gram_text, RIGHT)
        
        display_matrix_t.set_row_colors(GREEN, RED)
        
        
        display_matrix = Matrix(matrix).scale(0.9).next_to(display_matrix_t, RIGHT)
        
        display_matrix.set_column_colors(GREEN, RED)
        
        equals = MathTex(r"=", color=BLUE).scale(0.9).next_to(display_matrix, RIGHT)
        
        display_gram = Matrix(gram_matrix).scale(0.9).next_to(equals, RIGHT)
        
        entries = display_gram.get_entries()
        
        colors = [GREEN, RED]
        
        for row in range(2):
            for col in range(2):
                entries[col + 2*row].set_color(colors[row] if row==col else YELLOW)
        
        gram_group = VGroup(gram_text, display_matrix_t, display_matrix, equals, display_gram)
        
        gram_group.next_to(coord_text, DOWN)
        
        self.play(Write(gram_text), Create(display_matrix_t), Create(display_matrix))
        self.wait()
        self.play(Write(equals), Create(display_gram))
        self.wait()
        
        small_gram_group = VGroup(gram_text, display_gram)
        
        self.play(FadeOut(display_matrix_t), FadeOut(display_matrix), FadeOut(equals), display_gram.animate.next_to(gram_text, RIGHT))
        self.play(small_gram_group.animate.scale(1/0.9).next_to(coord_text, DOWN))
        self.wait()
        
        temp = display_gram.copy().next_to(display_gram, DOWN)
        
        det_text = get_det_text(temp, determinant="32 = 5.66^2", initial_scale_factor=1)
        det_text.set_color(BLUE)
        
        gram_det_text = MathTex(r"\det (\text{ Gramian }) = ", color=BLUE).next_to(det_text, LEFT)
        
        self.play(Write(gram_det_text), Create(temp), Write(det_text[:3]))
        self.wait()
        self.play(Write(det_text[3:]))
        self.wait()
        

        

class WhatWeCanDoSoFar3(Scene):
    def construct(self):
        title_text = Tex("What we know so far").scale(2)
        title_text.to_edge(UP)
        
        self.add(title_text)
    
        determinant_text = Tex(r"Given $n$ vectors in $\mathbb{R}^n$ $\longrightarrow$ ", "take the determinant").shift(UP)
        determinant_text[1].set_color(GREEN)
        
        self.add(determinant_text)
        
        cross_text = Tex(r"Given 2 vectors in $\mathbb{R}^3$ $\longrightarrow$ ", "magnitude of cross product")
        cross_text[1].set_color(GREEN)
        self.add(cross_text)
        
        box = SurroundingRectangle(cross_text)
        self.play(Create(box))
        self.wait()
        
        question_text = Tex(r"Is this consistent with Gramian?", color=YELLOW).next_to(cross_text, DOWN).shift(DOWN)
        
        self.play(Write(question_text))
        self.wait()



class LinkToCrossProduct(Scene):
    def construct(self):
        scalar_triple_eq = MathTex(r"\vec{\mathbf{a} } \cdot (\vec{\mathbf{b} } \times \vec{\mathbf{c} }) = \vec{\mathbf{b} } \cdot (\vec{\mathbf{c} } \times \vec{\mathbf{a} })").shift(UP)
        
        vector_triple_eq = MathTex(r"\vec{\mathbf{a} } \times (\vec{\mathbf{b} } \times \vec{\mathbf{c} }) = (\vec{\mathbf{a} } \cdot \vec{\mathbf{c} })\vec{\mathbf{b} } - (\vec{\mathbf{a} } \cdot \vec{\mathbf{b} })\vec{\mathbf{c} }").shift(DOWN)
        
        self.play(Write(scalar_triple_eq))
        self.wait()
        self.play(Write(vector_triple_eq))
        self.wait()
        self.play(FadeOut(scalar_triple_eq), FadeOut(vector_triple_eq))
        self.wait()
        
        cross_eq = MathTex(r"(", r"\vec{\mathbf{a} }", r"\times", r"\vec{\mathbf{b} }", r") \cdot(", r"\vec{\mathbf{a} }", r"\times", r"\vec{\mathbf{b} }", r")", r" = ", r"\vec{\mathbf{a} }", r"\cdot (", r"\vec{\mathbf{b} }", r"\times", r"(", r"\vec{\mathbf{a} }", r"\times", r"\vec{\mathbf{b} }", r"))").scale(1.5).to_edge(UP)
        cross_eq.set_color_by_tex("{a} }", GREEN)
        cross_eq.set_color_by_tex("{b} }", BLUE)
        
        self.play(Write(cross_eq))
        self.wait()
        
        line_2 = MathTex(r"=", r"\vec{\mathbf{a} }", r"\cdot ((", r"\vec{\mathbf{b} }", r"\cdot", r"\vec{\mathbf{b} }", r")", r"\vec{\mathbf{a} }", r"- (", r"\vec{\mathbf{b} }", r"\cdot", r"\vec{\mathbf{a} }", r")", r"\vec{\mathbf{b} }", r")").next_to(cross_eq, DOWN).scale(1.5).align_to(cross_eq, LEFT)
        line_2.set_color_by_tex("{a} }", GREEN)
        line_2.set_color_by_tex("{b} }", BLUE)
        
        self.play(Write(line_2))
        self.wait()
        
        line_3 = MathTex(r"= (", r"\vec{\mathbf{a} }", r"\cdot", r"\vec{\mathbf{a} }", r")(", r"\vec{\mathbf{b} }", r"\cdot", r"\vec{\mathbf{b} }", r") - (", r"\vec{\mathbf{b} }", r"\cdot", r"\vec{\mathbf{a} }", r")(", r"\vec{\mathbf{a} }", r"\cdot", r"\vec{\mathbf{b} }", r")").scale(1.5).next_to(line_2, DOWN).align_to(line_2, LEFT)
        
        line_3.set_color_by_tex("{a} }", GREEN)
        line_3.set_color_by_tex("{b} }", BLUE)
        
        self.play(Write(line_3))
        self.wait()
        
        names = ["a", "b"]
        
        mat_entries = list()
        for i in range(2):
            mat_entries.append([r"\vec{\mathbf{%s} } \cdot \vec{\mathbf{%s} }" % (names[i], names[j]) for j in range(2)])
        
        example_matrix = Matrix(mat_entries).scale(1.5)
        
        colors = [GREEN, BLUE]
        
        mat_entries = example_matrix.get_entries()
        
        for i in range(2):
            for j in range(2):
                mat_entries[i + 2*j][0][:2].set_color(colors[j]) # First vector
                mat_entries[i + 2*j][0][3:].set_color(colors[i]) # Second vector
        
        det_text = get_det_text(example_matrix, initial_scale_factor=1)
        
        equals = MathTex(r"=").scale(1.5).next_to(det_text, LEFT)
        
        det_grp = VGroup(equals, det_text, example_matrix).next_to(line_3, DOWN).align_to(line_3, LEFT)
        
        self.play(Write(det_grp))
        self.wait()




class LinearMapToMatrix2Dto3D(Scene):
    def construct(self):
        array_v = [1, -2, 1]
        array_w = [2, 0, 2]
        
        matrix = np.transpose([array_v, array_w])
        
        display_matrix = Matrix(matrix).scale(1.5)
        display_matrix.set_column_colors(GREEN, RED)
        
        self.play(Create(display_matrix))
        self.wait()
        
        title_text = Tex(r"What transformation does this represent?", color=YELLOW).to_edge(UP).scale(1.5)
        
        self.play(Write(title_text))
        self.wait()
        
        i_brace = Brace(display_matrix[0][4], DOWN)
        i_brace_text = i_brace.get_text(r"Where ", r"$\hat{\imath}$", r" goes").scale(0.7).shift(LEFT*0.1)
        i_brace_text[1].set_color(GREEN)
        
        j_brace = Brace(display_matrix[0][5], DOWN)
        j_brace_text = j_brace.get_text(r"Where ", r"$\hat{\jmath}$", r" goes").scale(0.7)
        j_brace_text[1].set_color(RED)
        
        self.play(Create(i_brace), Write(i_brace_text))
        self.wait()
        self.play(Create(j_brace), Write(j_brace_text))
        self.wait()


class LinearMapToMatrix2Dto3DVisual(ThreeDScene):

    ###### Stolen from why determinant should exist
    
    def add_square_custom(self, size, left_corner):
        square = Rectangle(
            color=YELLOW,
            width=size,
            height=size,
            stroke_color=YELLOW,
            stroke_width=3,
            fill_color=YELLOW,  
            fill_opacity=0.3,
        )
        square.move_to(self.grid.coords_to_point(*left_corner), DL)
        return square
    
    def approximate_circ_with_num(self, num):
        square_size = 2/num
        squares = []
        for left_edge in range(-1*num, 1*num):
            for bottom_edge in range(-1*num, 1*num):
                # Check if in circle or not
                center_x = (left_edge + 0.5) * square_size
                center_y = (bottom_edge + 0.5) * square_size
                if (center_x**2 + center_y**2) <= 1:
                    squares.append(self.add_square_custom(square_size, [left_edge * square_size, bottom_edge * square_size]))
        return VGroup(*squares)

    def construct(self):
        array_v = [1, -2, 1]
        array_w = [2, 0, 2]
        temp_array = [0,0,0]
        
        linear_text = Tex(r"Linear $\Leftrightarrow$ gridlines remain parallel and evenly spaced", color=YELLOW).to_edge(UP).add_background_rectangle()
        
        matrix = np.transpose([array_v, array_w, temp_array])
        axes = ThreeDAxes(z_min=-4.5, z_max=4.5)
        
        self.add(axes)
        
        self.grid = NumberPlane()
        grid = self.grid
        self.add(grid)
        
        i_hat = grid.get_vector([1, 0, 0], color=GREEN)
        j_hat = grid.get_vector([0, 1, 0], color=RED)
        
        vector_v = Arrow3D([0, 0, 0], array_v, color=GREEN)
        vector_w = Arrow3D([0, 0, 0], array_w, color=RED)
        
        self.play(GrowArrow(i_hat), GrowArrow(j_hat))
        self.wait()
        
        self.move_camera(phi=60 * DEGREES, theta=-100 * DEGREES)
        self.wait()
        
        self.play(Create(vector_v), Create(vector_w))
        
        grid_grp = VGroup(grid, i_hat, j_hat)
        
        grid_grp.save_state()
        
        self.add_fixed_in_frame_mobjects(linear_text)
        self.play(Write(linear_text[1]))
        self.wait()
        
        self.play(ApplyMatrix(matrix, grid_grp))
        self.wait()
        
        ### Now return to 2D
        
        self.play(FadeOut(linear_text))
        self.play(FadeOut(vector_v), FadeOut(vector_w))
        self.play(Restore(grid_grp))
        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES)
        self.play(FadeOut(i_hat), FadeOut(j_hat))
        self.wait()
        
        i_hat.set_opacity(0)
        j_hat.set_opacity(0)
        
        #### Time to add squares
        
        square_1 = self.add_square_custom(1, [0, 0])
        
        self.play(DrawBorderThenFill(square_1))
        self.wait()
        
        grid_grp = VGroup(grid, square_1)
        grid_grp.save_state()
        
        self.move_camera(phi=60 * DEGREES, theta=-100 * DEGREES)
        self.play(ApplyMatrix(matrix, grid_grp))
        self.wait()
        
        area_text_1 = Tex(r"Area scaled by $k$").to_edge(UP).add_background_rectangle()
        self.add_fixed_in_frame_mobjects(area_text_1)
        self.play(Write(area_text_1[1]))
        self.wait()
        self.play(FadeOut(area_text_1))
        self.play(Restore(grid_grp))
        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES)
        self.wait()
        
        # MOAR squares
        
        square_2 = self.add_square_custom(1, [-2, 1])
        square_3 = self.add_square_custom(0.5, [2, 0])
        square_4 = self.add_square_custom(2, [0, -3])
        square_5 = self.add_square_custom(0.2, [-1, 0])
        self.play(DrawBorderThenFill(square_2), DrawBorderThenFill(square_3), DrawBorderThenFill(square_4), DrawBorderThenFill(square_5))
        self.wait()
        
        squares = VGroup(square_1, square_2, square_3, square_4, square_5)
        
        grid_grp = VGroup(grid, squares)
        grid_grp.save_state()
        
        self.move_camera(phi=60 * DEGREES, theta=-100 * DEGREES)
        self.play(ApplyMatrix(matrix, grid_grp))
        self.wait()
        
        area_text_2 = Tex(r"All grid squares scaled by $k$").to_edge(UP).add_background_rectangle()
        self.add_fixed_in_frame_mobjects(area_text_2)
        self.play(Write(area_text_2[1]))
        self.wait()
        self.play(FadeOut(area_text_2))
        self.play(Restore(grid_grp))
        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES)
        self.play(FadeOut(squares))
        self.wait()
        
        # Circle time
        
        test_circ = Circle(fill_color=RED,  
            fill_opacity=0.3,)
        self.play(Create(test_circ))
        
        squares_1 = self.approximate_circ_with_num(5)
        self.play(Create(squares_1))
        self.play(FadeOut(squares_1))
        squares_2 = self.approximate_circ_with_num(10)
        self.play(Create(squares_2))
        self.play(FadeOut(squares_2))
        
        squares_3 = self.approximate_circ_with_num(20)
        self.play(Create(squares_3))
        
        grid_grp = VGroup(grid, squares_3, test_circ)
        
        self.move_camera(phi=60 * DEGREES, theta=-100 * DEGREES)
        self.play(ApplyMatrix(matrix, grid_grp))
        self.wait()
        
        area_text_3 = Tex(r"Arbitrary region scaled by $k$", color=YELLOW).to_edge(UP).add_background_rectangle()
        
        self.add_fixed_in_frame_mobjects(area_text_3)
        self.play(Write(area_text_3[1]))
        self.wait()
        
class ExampleGramianasAreaScaleFactor(Scene):
    def construct(self):
        array_v = [1, -2, 1]
        array_w = [2, 0, 2]
        
        gram_text = Tex(r"Gramian $=$", color=BLUE).scale(0.9).to_edge(LEFT)
        
        matrix = np.transpose([np.transpose(array_v), np.transpose(array_w)])
        
        gram_matrix = np.matmul(np.transpose(matrix), matrix)
        
        display_matrix_t = Matrix(np.transpose(matrix)).scale(0.9).next_to(gram_text, RIGHT)
        
        display_matrix_t.set_row_colors(GREEN, RED)
        
        
        display_matrix = Matrix(matrix).scale(0.9).next_to(display_matrix_t, RIGHT)
        
        display_matrix.set_column_colors(GREEN, RED)
        
        equals = MathTex(r"=", color=BLUE).scale(0.9).next_to(display_matrix, RIGHT)
        
        display_gram = Matrix(gram_matrix).scale(0.9).next_to(equals, RIGHT)
        
        entries = display_gram.get_entries()
        
        colors = [GREEN, RED]
        
        for row in range(2):
            for col in range(2):
                entries[col + 2*row].set_color(colors[row] if row==col else YELLOW)
        
        gram_group = VGroup(gram_text, display_matrix_t, display_matrix, equals, display_gram)
        
        
        self.play(Write(gram_text), Create(display_matrix_t), Create(display_matrix))
        self.wait()
        self.play(Write(equals), Create(display_gram))
        self.wait()
        self.play(gram_group.animate.to_edge(UP))
        self.wait()

        temp = display_gram.copy().next_to(display_gram, DOWN)
        
        det_text = get_det_text(temp, determinant="32 = 5.66^2", initial_scale_factor=1)
        det_text.set_color(BLUE)
        
        gram_det_text = MathTex(r"\det (\text{ Gramian }) = ", color=BLUE).next_to(det_text, LEFT)
        
        det_grp = VGroup(gram_det_text, temp, det_text)
        det_grp.shift(DOWN).to_edge(LEFT)
        
        self.play(Write(gram_det_text), Create(temp), Write(det_text[:3]))
        self.wait()
        self.play(Write(det_text[3:]))
        self.wait()
        
        sqrt_text = MathTex(r"\sqrt{\det (\text{ Gramian }) } = ", r"5.66", color=BLUE).next_to(det_grp, DOWN)
        
        self.play(Write(sqrt_text))
        self.wait()
        
        scale_brace = Brace(sqrt_text, DOWN)
        scale_brace_text = scale_brace.get_text(r"Area scale factor")
        scale_brace_text.set_color(YELLOW)
        
        self.play(Create(scale_brace), Write(scale_brace_text))
        self.wait()
        


class WhatGramDeterminantIsSaying(Scene):
    def construct(self):
        
        theorem_text = Tex(r'"Linear volume scaling theorem"', color=YELLOW).scale(1.8).to_edge(UP)
        
        self.play(Write(theorem_text))
        self.wait()
        
        note_text = Tex(r"(Not sure if this actually has a name, just a temp name)").scale(0.7).next_to(theorem_text, DOWN)
        
        self.play(Write(note_text))
        self.wait()
        
        linear_text = Tex(r"Linear map from $\mathbb{R}^n$ to $\mathbb{R}^d$").scale(1.4).next_to(note_text, DOWN).shift(DOWN*0.5)
        
        self.play(Write(linear_text))
        self.wait()
        
        scale_text = Tex(r"scales $n$-dimensional volume by a constant", color=YELLOW).scale(1.4).next_to(linear_text, DOWN)
        
        self.play(Write(scale_text))
        self.wait()
        
        line_1 = Tex(r"If $n = d$, ", r"use determinant").next_to(scale_text, DOWN).to_edge(LEFT)
        line_1[1].set_color(GREEN)
        
        self.play(Write(line_1))
        self.wait()
        
        line_2 = Tex(r"If $n < d$, ", r"use square root of gramian determinant").next_to(line_1, DOWN).to_edge(LEFT)
        line_2[1].set_color(GREEN)
        
        self.play(Write(line_2))
        self.wait()
        
        line_3 = Tex(r"If $n > d$, ", r"it's $0$, since squish into lower dimension").next_to(line_2, DOWN).to_edge(LEFT)
        line_3[1].set_color(GREEN)
        
        self.play(Write(line_3))
        self.wait()
        

class NowMovingToMatrixItself(Scene):
    def construct(self):
        focus_1 = Tex(r"Been focusing on gramian determinant").scale(1.4).shift(UP)
        self.play(Write(focus_1))
        self.wait()
        
        focus_2 = Tex(r"Now onto gramian matrix", color=YELLOW).scale(2).shift(DOWN)
        
        self.play(Write(focus_2))
        self.wait()
      

class WhatTheGramianIs(Scene):
    def construct(self):
        what_text = Tex(r"Gramian is matrix of ", r"dot products").scale(1.5).to_edge(UP)
        what_text[1].set_color(YELLOW)
        
        names = ["a", "b", "c"]
        
        mat_entries = list()
        for i in range(3):
            mat_entries.append([r"\vec{\mathbf{%s} } \cdot \vec{\mathbf{%s} }" % (names[i], names[j]) for j in range(3)])
        
        example_matrix = Matrix(mat_entries)
        
        colors = [RED, GREEN, BLUE]
        
        mat_entries = example_matrix.get_entries()
        
        for i in range(3):
            for j in range(3):
                mat_entries[i + 3*j][0][:2].set_color(colors[j]) # First vector
                mat_entries[i + 3*j][0][3:].set_color(colors[i]) # Second vector
                
        
        
        self.play(Write(what_text))
        self.wait()
        
        self.play(Create(example_matrix))
        self.wait()


  


class InvariantUnderRotation(Scene):
    def construct(self):
        names = ["a", "b", "c"]
        
        mat_entries = list()
        for i in range(3):
            mat_entries.append([r"\vec{\mathbf{%s} } \cdot \vec{\mathbf{%s} }" % (names[i], names[j]) for j in range(3)])
        
        example_matrix = Matrix(mat_entries)
        
        colors = [RED, GREEN, BLUE]
        
        mat_entries = example_matrix.get_entries()
        
        for i in range(3):
            for j in range(3):
                mat_entries[i + 3*j][0][:2].set_color(colors[j]) # First vector
                mat_entries[i + 3*j][0][3:].set_color(colors[i]) # Second vector
                
        self.play(Create(example_matrix))
        self.wait()
        
        invar_text = Tex(r"Invariant under an orthogonal matrix").to_edge(UP)
        
        stronger_text = Tex(r"Something stronger...", color=YELLOW).next_to(invar_text, DOWN)
        
        self.play(Write(invar_text))
        self.wait()
        self.play(Write(stronger_text))
        self.wait()

class RandomVectors(Scene):
    def construct(self):
        axes = ThreeDAxes(z_min=-4.5, z_max=4.5)
        array_v = [1, -2, 1]
        array_w = [2, 0, 2]
        mid_axis = normalize([-1, 0, 1])
        rotate_matrix = rotation_matrix(-60 * DEGREES, mid_axis)
        
        array_1 = [1, 3, 0]
        array_2 = [1, 1, -1]
        array_3 = [2, 4, -1]
        
        array_4 = np.transpose(np.matmul(rotate_matrix, np.transpose(array_1)))
        array_5 = np.transpose(np.matmul(rotate_matrix, np.transpose(array_2)))
        array_6 = np.transpose(np.matmul(rotate_matrix, np.transpose(array_3)))
        
        arrays = [array_1, array_2, array_3, array_4, array_5, array_6]
        vectors = list()
        
        for array, color in zip(arrays, [BLUE, BLUE, BLUE, PURPLE, PURPLE, PURPLE]):
            vectors.append(DecimalMatrix([[elem] for elem in array], element_to_mobject_config={'num_decimal_places': 2}).set_color(color))
        
        
        vectors_1 = VGroup(*vectors[:3]).arrange(RIGHT).to_edge(UP).to_edge(LEFT)
        vectors_2 = VGroup(*vectors[3:]).arrange(RIGHT).to_edge(UP).to_edge(RIGHT)
        vectors = VGroup(*vectors).to_edge(UP)
        
        self.play(Create(vectors_1))
        self.wait()
        self.play(Create(vectors_2))
        self.wait()
        
        matrix = np.transpose([np.transpose(array_1), np.transpose(array_2), np.transpose(array_3)])
        
        common_gramian = DecimalMatrix(np.matmul(np.transpose(matrix), matrix), element_to_mobject_config={'num_decimal_places':2}).set_color(YELLOW)
        
        common_gramian_text = Tex(r"Common Gramian $=$ ", color=YELLOW).next_to(common_gramian, LEFT)
        
        common_gramian_grp = VGroup(common_gramian, common_gramian_text)
        common_gramian_grp.next_to(vectors, DOWN)
        
        self.play(Write(common_gramian_text), Create(common_gramian))
        self.wait()
        

class RandomVectorsVisual(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(z_min=-4.5, z_max=4.5)
        array_v = [1, -2, 1]
        array_w = [2, 0, 2]
        mid_axis = normalize([-1, 0, 1])
        rotate_matrix = rotation_matrix(-60 * DEGREES, mid_axis)
        
        array_1 = np.array([1, 3, 0])
        array_2 = np.array([1, 1, -1])
        array_3 = np.array([2, 4, -1])
        
        array_4 = np.transpose(np.matmul(rotate_matrix, np.transpose(array_1)))
        array_5 = np.transpose(np.matmul(rotate_matrix, np.transpose(array_2)))
        array_6 = np.transpose(np.matmul(rotate_matrix, np.transpose(array_3)))
        
        arrays = [array_1, array_2, array_3, array_4, array_5, array_6]
        vectors = list()
        
        for array, color in zip(arrays, [BLUE, BLUE, BLUE, PURPLE, PURPLE, PURPLE]):
            vectors.append(Arrow3D([0, 0, 0], array, color=color))
        
        vectors = VGroup(*vectors)
        
        self.set_camera_orientation(phi=60 * DEGREES, theta= 80 * DEGREES)
        
        self.add(axes)
        self.play(Create(vectors))
        #self.begin_ambient_camera_rotation(rate=0.2)
        self.wait()
        
        self.play(Rotate(vectors[:3], axis=mid_axis, about_point=[0,0,0], angle=-60 * DEGREES))
        self.wait()


class HowWeMightProveThis0(Scene):
    def construct(self):
        title_text = Tex(r"How to prove this?").scale(2).to_edge(UP)
        
        self.play(Write(title_text))
        self.wait()
        
        line_1 = MathTex(r"\text{Got }", r"\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_n\}", r",", r"\{\vec{\mathbf{v} }_1, \dots, \vec{\mathbf{v} }_n\}", r"\text{ in }", r"\mathbb{R}^d").next_to(title_text, DOWN)
        line_1[1].set_color(BLUE)
        line_1[3].set_color(PURPLE)
        
        self.play(Write(line_1))
        self.wait()
        
        line_2 = MathTex(r"\text{Know }", r"\vec{\mathbf{u} }_i \cdot \vec{\mathbf{u} }_j", r"=", r"\vec{\mathbf{v} }_i \cdot \vec{\mathbf{v} }_j").next_to(line_1, DOWN)
        line_2[1].set_color(BLUE)
        line_2[3].set_color(PURPLE)
        
        self.play(Write(line_2))
        self.wait()
        
        line_3 = Tex(r"Want to say: rotate down to $n$-dimensional, use change of basis").scale(0.9).next_to(line_2, DOWN)
        
        self.play(Write(line_3))
        self.wait()
        
        problem_text = Tex(r"Problem: might be linearly dependent").set_color(RED).next_to(line_3, DOWN)
        
        self.play(Write(problem_text))
        self.wait()
        

class SameGramianLinearlyDependent(VectorScene):
    def construct(self):
        self.plane = self.add_plane()
        
        u_1 = self.add_vector([2, 0, 0], color=BLUE)
        u_2 = self.add_vector([0, 1, 0], color=BLUE)
        u_3 = self.add_vector([2, 1, 0], color=BLUE)
        self.wait()
        
        u_1_label = MathTex(r"\vec{\mathbf{u} }_1", color=BLUE).next_to(u_1, RIGHT + DOWN)
        u_2_label = MathTex(r"\vec{\mathbf{u} }_2", color=BLUE).next_to(u_2, UP + LEFT)
        u_3_label = MathTex(r"\vec{\mathbf{u} }_3", color=BLUE).next_to(u_3, UP + RIGHT)
        
        self.play(Write(u_1_label), Write(u_2_label), Write(u_3_label))
        self.wait()
        
        v_1 = self.add_vector([-2, 0, 0], color=PURPLE)
        v_2 = self.add_vector([0, -1, 0], color=PURPLE)
        v_3 = self.add_vector([-2, -1, 0], color=PURPLE)
        self.wait()
        
        v_1_label = MathTex(r"\vec{\mathbf{v} }_1", color=PURPLE).next_to(v_1, LEFT + UP)
        v_2_label = MathTex(r"\vec{\mathbf{v} }_2", color=PURPLE).next_to(v_2, DOWN + RIGHT)
        v_3_label = MathTex(r"\vec{\mathbf{v} }_3", color=PURPLE).next_to(v_3, DOWN + LEFT)
        
        self.play(Write(v_1_label), Write(v_2_label), Write(v_3_label))
        self.wait()
        
        same_text = Tex(r"Same Gramian - differ by rotation").set_color(YELLOW).to_edge(UP).add_background_rectangle()
        
        self.add(same_text)
        self.play(Write(same_text[1]))
        self.wait()
        self.play(Rotate(VGroup(u_1, u_2, u_3), about_point=[0, 0, 0]))
        self.wait()
        self.play(Rotate(VGroup(u_1, u_2, u_3), about_point=[0, 0, 0]))
        self.wait()
        
        lin_text = Tex(r"Linearly dependent, so can't use change of basis", color=RED).next_to(same_text, DOWN).add_background_rectangle()
        
        self.add(lin_text)
        self.play(Write(lin_text[1]))
        self.wait()
        
        self.play(FadeOut(same_text), FadeOut(lin_text))
        self.wait()


class CorrespondingLinearRelations(Scene):
    def construct(self):
        title_text = Tex(r"Key idea: ", r"corresponding ", r"linear relations").scale(1.5).to_edge(UP)
        title_text[1:].set_color(YELLOW)
        
        info_brace = Brace(title_text[2], DOWN)
        info_brace_text = info_brace.get_text(r"Linear combination equal to $\vec{\mathbf{0} }$").to_edge(RIGHT)
        
        self.play(Write(title_text))
        self.wait()
        self.play(Create(info_brace), Write(info_brace_text))
        self.wait()
        self.play(FadeOut(info_brace), FadeOut(info_brace_text))
        self.wait()
        
        key_property = MathTex(r"\vec{\mathbf{u} }_i \cdot \vec{\mathbf{u} }_j", r"=", r"\vec{\mathbf{v} }_i \cdot \vec{\mathbf{v} }_j").scale(1.5).next_to(title_text, DOWN).to_edge(LEFT)
        key_property[0].set_color(BLUE)
        key_property[2].set_color(PURPLE)
        
        self.play(Write(key_property))
        self.wait()
        
        line_1 = MathTex(r"\sum_i \lambda_i \vec{\mathbf{u} }_i", r" = \vec{\mathbf{0} }").scale(0.9).next_to(title_text, DOWN)
        line_1[0].set_color(BLUE)
        
        self.play(Write(line_1))
        self.wait()
        
        line_2 = MathTex(r"\Leftrightarrow ", r"(\sum_i \lambda_i \vec{\mathbf{u} }_i) \cdot (\sum_i \lambda_i \vec{\mathbf{u} }_i)", r" = 0").scale(0.9).next_to(line_1, DOWN)
        line_2[1].set_color(BLUE)
        
        self.play(Write(line_2))
        self.wait()
        
        length_brace = Brace(line_2, DOWN)
        length_brace_text = length_brace.get_text(r"Vector is $\vec{\mathbf{0} }$ iff its length is $0$")
        
        self.play(Create(length_brace), Write(length_brace_text))
        self.wait()
        self.play(FadeOut(length_brace), FadeOut(length_brace_text))
        self.wait()
        
        line_2_1 = MathTex(r"\Leftrightarrow ", r"(\sum_i \lambda_i \vec{\mathbf{u} }_i) \cdot (\sum_j \lambda_j \vec{\mathbf{u} }_j)", r" = 0").scale(0.9).next_to(line_1, DOWN)
        line_2_1[1].set_color(BLUE)
        
        self.play(Transform(line_2, line_2_1))
        self.wait()
        
        collision_brace = Brace(line_2[1], DOWN)
        collision_brace_text = collision_brace.get_text(r"Will avoid collisions in notation")
        
        self.play(Create(collision_brace), Write(collision_brace_text))
        self.wait()
        self.play(FadeOut(collision_brace), FadeOut(collision_brace_text))
        self.wait()
        
        line_3 = MathTex(r"\Leftrightarrow", r"\sum_{i, j} \lambda_i \lambda_j ", r"\vec{\mathbf{u} }_i \cdot \vec{\mathbf{u} }_j", r"= 0").scale(0.9).next_to(line_2, DOWN)
        line_3[1:3].set_color(BLUE)
        
        self.play(Write(line_3))
        self.wait()
        
        box_1 = SurroundingRectangle(line_3[2])
        self.play(Create(box_1))
        self.wait()
        
        box_2 = SurroundingRectangle(key_property)
        self.play(TransformFromCopy(box_1, box_2))
        self.wait()
        
        line_4 = MathTex(r"\Leftrightarrow", r"\sum_{i, j} \lambda_i \lambda_j ", r"\vec{\mathbf{v} }_i \cdot \vec{\mathbf{v} }_j", r"= 0").scale(0.9).next_to(line_3, DOWN)
        line_4[1:3].set_color(PURPLE)
        
        self.play(Write(line_4))
        self.wait()
        
        self.play(FadeOut(box_1), FadeOut(box_2))
        self.wait()
        
        line_5 = MathTex(r"\Leftrightarrow", r"\sum_i \lambda_i \vec{\mathbf{v} }_i", r" = \vec{\mathbf{0} }").scale(0.9).next_to(line_4, DOWN)
        line_5[1].set_color(PURPLE)
        
        self.play(Write(line_5))
        self.wait()
        
        self.play(FadeOut(line_2), FadeOut(line_3), FadeOut(line_4), FadeOut(key_property))
        self.play(line_1.animate.move_to(line_3.get_center() + LEFT))
        self.play(line_5.animate.next_to(line_1, RIGHT))
        self.play(VGroup(line_1, line_5).animate.move_to([0, 0, 0]).scale(2))
        self.wait()
        

class CanDeduceWithoutLooking(VectorScene):
    def construct(self):
        self.plane = self.add_plane()
        
        u_1 = self.add_vector([2, 0, 0], color=BLUE)
        u_2 = self.add_vector([0, 1, 0], color=BLUE)
        u_3 = self.add_vector([2, 1, 0], color=BLUE)
        self.wait()
        
        u_1_label = MathTex(r"\vec{\mathbf{u} }_1", color=BLUE).next_to(u_1, RIGHT + DOWN)
        u_2_label = MathTex(r"\vec{\mathbf{u} }_2", color=BLUE).next_to(u_2, UP + LEFT)
        u_3_label = MathTex(r"\vec{\mathbf{u} }_3", color=BLUE).next_to(u_3, UP + RIGHT)
        
        self.play(Write(u_1_label), Write(u_2_label), Write(u_3_label))
        self.wait()
        
        self.play(u_2.animate.shift(RIGHT * 2), u_2_label.animate.shift(RIGHT*2 + DOWN + RIGHT))
        self.wait()
        
        u_eq = MathTex(r"\vec{\mathbf{u} }_3 = \vec{\mathbf{u} }_1 + \vec{\mathbf{u} }_2", color=BLUE).scale(2).to_edge(UP).to_edge(LEFT).add_background_rectangle()
        self.add(u_eq)
        self.play(Write(u_eq[1]))
        self.wait()
        
        u_eq_2 = MathTex(r"\vec{\mathbf{u} }_3 - \vec{\mathbf{u} }_1 - \vec{\mathbf{u} }_2 = \vec{\mathbf{0} }", color=BLUE).scale(2).next_to(u_eq, DOWN).to_edge(LEFT).add_background_rectangle()
        self.add(u_eq_2)
        self.play(Write(u_eq_2[1]))
        self.wait()
        
        v_eq = MathTex(r"\Rightarrow \vec{\mathbf{v} }_3 - \vec{\mathbf{v} }_1 - \vec{\mathbf{v} }_2 = \vec{\mathbf{0} }", color=PURPLE).scale(2).next_to(u_eq_2, DOWN).to_edge(LEFT).add_background_rectangle()
        
        self.add(v_eq)
        self.play(Write(v_eq[1]))
        self.wait()
        
        v_eq_2 = MathTex(r"\Rightarrow \vec{\mathbf{v} }_3 = \vec{\mathbf{v} }_1 + \vec{\mathbf{v} }_2", color=PURPLE).scale(2).next_to(v_eq, DOWN).to_edge(LEFT).add_background_rectangle()
        self.add(v_eq_2)
        self.play(Write(v_eq_2[1]))
        self.wait()


class HowToFindThatIsometry(Scene):
    def construct(self):
        title_text = Tex(r"How to find the orthogonal matrix").scale(1.5).to_edge(UP).set_color(YELLOW)
        self.play(Write(title_text))
        self.wait()
        
        line_1 = MathTex(r"\text{Find }", r"\{\vec{\mathbf{u} }_1, \dots, \vec{\mathbf{u} }_k\}", r"\text{ as a span-basis}").next_to(title_text, DOWN).shift(DOWN)
        line_1[1].set_color(BLUE)
        
        line_2 = MathTex(r"\text{Implies }", r"\{\vec{\mathbf{v} }_1, \dots, \vec{\mathbf{v} }_k\}", r"\text{ is a span-basis}").next_to(line_1, DOWN)
        line_2[1].set_color(PURPLE)
        
        line_3_0 = Tex(r"Rotate both down to $k$ dimensions, then use change of basis").next_to(line_2, DOWN)
        line_3_1 = Tex(r"(fixing the other $d - k$ standard basis vectors)").next_to(line_3_0, DOWN)
        
        self.play(Write(line_1))
        self.wait()
        self.play(Write(line_2))
        self.wait()
        self.play(Write(line_3_0))
        self.play(Write(line_3_1))
        self.wait()
   
class GramianAsNumericalEncodingOfShape(Scene):
    def construct(self):
        like_text = Tex(r"Gramian ", r"$\Leftrightarrow$", r" numerical encoding of shape").scale(1.2).to_edge(UP)
        like_text[2].set_color(YELLOW)
        
        self.play(Write(like_text))
        self.wait()
        
        because_text = Tex(r"Encodes vectors ", r"\textit{independent of rotation/reflection}").next_to(like_text, DOWN)
        because_text[1].set_color(YELLOW)
        
        self.play(Write(because_text))
        self.wait()
        
        names = ["a", "b", "c"]
        
        mat_entries = list()
        for i in range(3):
            mat_entries.append([r"\vec{\mathbf{%s} } \cdot \vec{\mathbf{%s} }" % (names[i], names[j]) for j in range(3)])
        
        example_matrix = Matrix(mat_entries)
        
        colors = [RED, GREEN, BLUE]
        
        mat_entries = example_matrix.get_entries()
        
        for i in range(3):
            for j in range(3):
                mat_entries[i + 3*j][0][:2].set_color(colors[j]) # First vector
                mat_entries[i + 3*j][0][3:].set_color(colors[i]) # Second vector
        
        example_matrix
        
        self.play(Create(example_matrix))
        self.wait()
        
        self.play(example_matrix.animate.to_edge(LEFT))
        
        entry_text = Tex(r"Entries $\Leftrightarrow$ ", r"lengths and angles")
        entry_text[1].set_color(YELLOW)
        entry_text.next_to(example_matrix, RIGHT)
        
        self.play(Write(entry_text))
        self.wait()
        
        self.play(entry_text.animate.align_to(example_matrix, UP))
        
        intuitive_text = Tex(r"Intuitive - ", r"lengths and angles determine shape").scale(0.85)
        intuitive_text[1].set_color(YELLOW)
        
        intuitive_text.next_to(example_matrix, RIGHT)
        
        self.play(Write(intuitive_text))
        self.wait()
        
        nonobvious_text = Tex(r"Not obvious - ", r"determinant links to volume").scale(0.85)
        nonobvious_text[1].set_color(GREEN)
        
        nonobvious_text.next_to(example_matrix, RIGHT).align_to(example_matrix, DOWN)
        
        self.play(Write(nonobvious_text))
        self.wait()



        
class LinkToDotProduct(Scene):
    def construct(self):
        compute_text = Tex(r"Computing dot product in (not necessarily orthonormal) basis $\{\vec{\mathbf{f_i} }\}$").scale(0.9).to_edge(UP)
        
        self.play(Write(compute_text))
        self.wait()
        
        line_1 = MathTex(r"(", r"x_i", r"\vec{\mathbf{f_i} }", r") \cdot (", r"y_j", r"\vec{\mathbf{f_j} }", r")", r"=", r"x_i", r"y_j", r"(", r"\vec{\mathbf{f_i} }", r"\cdot", r"\vec{\mathbf{f_j} }", r")"). next_to(compute_text, DOWN)
        line_1.set_color_by_tex("x_i", BLUE)
        line_1.set_color_by_tex("y_j", GREEN)
        line_1.set_color_by_tex("\mathbf{f", PURPLE)
        
        self.play(Write(line_1))
        self.wait()
        
        line_2 = MathTex(r"=", r"x_i", r"G_{ij}", r"y_j").next_to(line_1, DOWN).align_to(line_1, LEFT)
        line_2[1].set_color(BLUE)
        line_2[2].set_color(PURPLE)
        line_2[3].set_color(GREEN)
        
        self.play(Write(line_2))
        self.wait()
        
        line_3 = MathTex(r"=", r"\vec{\mathbf{x} }^T", r"\mathbf{G}", r"\vec{\mathbf{y} }").scale(1.5).next_to(line_2, DOWN).align_to(line_2, LEFT)
        
        line_3[1].set_color(BLUE)
        line_3[2].set_color(PURPLE)
        line_3[3].set_color(GREEN)
        
        self.play(Write(line_3))
        self.wait()
        
        comp_text = Tex(r"Gramian allows computing dot product in ", r"arbitrary basis").align_to(line_3.get_corner(DL), UP, alignment_vect=UP).shift(DOWN*0.5)
        comp_text[1].set_color(YELLOW)
        
        self.play(Write(comp_text))
        self.wait()
        
        measure_text = Tex(r"Measure", r" dot product $\Rightarrow$ called a ", r"Metric Matrix/Tensor").next_to(comp_text, DOWN).align_to(comp_text, LEFT)
        measure_text[0].set_color(YELLOW)
        measure_text[2].set_color(YELLOW)
        self.play(Write(measure_text))
        self.wait()
        self.play(FadeOut(measure_text))
        self.wait()
        
        
        definite_eq = MathTex(r"\vec{\mathbf{x} }^T", r"\mathbf{G}", r"\vec{\mathbf{x} }", r"=", r"|", r"x_i", r"\vec{\mathbf{e_i} }", r"|^2 \geq 0").next_to(comp_text, DOWN).align_to(comp_text, LEFT)
        definite_eq[0].set_color(BLUE)
        definite_eq[1].set_color(PURPLE)
        definite_eq[2].set_color(BLUE)
        definite_eq[5].set_color(BLUE)
        definite_eq[6].set_color(PURPLE)
        
        self.play(Write(definite_eq))
        self.wait()
        
        prop_text = Tex(r"Positive-(semi)definite", r" (and ", r"symmetric", r")").next_to(definite_eq, DOWN).align_to(comp_text, LEFT)
        prop_text[0].set_color(YELLOW)
        prop_text[2].set_color(YELLOW)
        
        self.play(Write(prop_text))
        self.wait()
        
        stronger_text = Tex(r"Something stronger...", color=YELLOW).next_to(prop_text, DOWN).align_to(comp_text, LEFT)
        
        self.play(Write(stronger_text))
        


class SymmetricAndPositiveDefiniteIsGramian(Scene):
    def construct(self):
        intro_text = Tex(r"Let ", r"$\mathbf{M}$", r" be a real, symmetric, positive-(semi)definite matrix").to_edge(UP)
        intro_text[1].set_color(PURPLE)
        
        self.play(Write(intro_text))
        self.wait()
        
        line_1 = MathTex(r"\mathbf{M}", r"=", r"\mathbf{P}", r"\mathbf{D}", r"\mathbf{P}^T", r",\space \space \mathbf{P}^T = \mathbf{P}^{-1}").next_to(intro_text, DOWN).shift(DOWN)
        line_1[0].set_color(PURPLE)
        line_1[2].set_color(GREEN)
        line_1[3].set_color(PURPLE)
        line_1[4:].set_color(GREEN)
        
        self.play(Write(line_1))
        self.wait()
        
        line_2_1 = Tex(r"Symmetric $\Rightarrow$ Real Eigenvalues")
        line_2_2 = Tex(r"Positive-(semi)definite $\Rightarrow$ Nonnegative Eigenvalues")
        
        line_2 = VGroup(line_2_1, line_2_2).next_to(line_1, DOWN).arrange(DOWN)
        
        self.play(Write(line_2))
        self.wait()
        
        line_3 = MathTex(r"\mathbf{U}", r"=", r"\mathbf{P}", r"\sqrt{\mathbf{D} }", r"\mathbf{P}^T", r" \Rightarrow", r"\mathbf{U}^T \mathbf{U}", r"=", r"\mathbf{M}").next_to(line_2, DOWN)
        line_3[0].set_color(BLUE)
        line_3[2].set_color(GREEN)
        line_3[3].set_color(PURPLE)
        line_3[4].set_color(GREEN)
        line_3[6].set_color(BLUE)
        line_3[8].set_color(PURPLE)
        
        self.play(Write(line_3))
        self.wait()
        
        line_4 = Tex(r"$\mathbf{M}$", r" is a ", r"Gramian Matrix", r", with ", r"vector realization", r" $\mathbf{U}$").next_to(line_3, DOWN)
        line_4[0].set_color(PURPLE)
        line_4[2].set_color(YELLOW)
        line_4[4].set_color(YELLOW)
        line_4[5].set_color(BLUE)
        
        self.play(Write(line_4))
        self.wait()


class SymmetricPositiveDefiniteExample(Scene):
    def construct(self):
        array_1 = [1, 0, 0]
        array_2 = [0, 3, 0]
        array_3 = [-1, -1, 1]
        
        matrix = np.transpose([np.transpose(array_1), np.transpose(array_2), np.transpose(array_3)])
        
        numerical_gramian = np.matmul(np.transpose(matrix), matrix)
        
        gramian = DecimalMatrix(numerical_gramian, element_to_mobject_config={'num_decimal_places':2}).set_color(PURPLE)
        
        g_eq = MathTex(r"\mathbf{G} = ", color=PURPLE).next_to(gramian, LEFT)
        
        gram_grp = VGroup(g_eq, gramian)
        
        self.play(Create(gramian), Write(g_eq))
        self.wait()
        self.play(gram_grp.animate.to_edge(UP))
        
        
        evals, matrix_P = np.linalg.eigh(numerical_gramian)
        
        matrix_D = np.diag(evals)
        
        decomp_eq = MathTex(r"\mathbf{G}", r"=", r"\mathbf{P}", r"\mathbf{D}", r"\mathbf{P}^T", r"\space \text{,with}").to_edge(LEFT)
        decomp_eq[0].set_color(PURPLE)
        decomp_eq[2].set_color(GREEN)
        decomp_eq[3].set_color(PURPLE)
        decomp_eq[4].set_color(GREEN)
        
        self.play(Write(decomp_eq))
        
        p_eq = MathTex(r"\mathbf{P} = ", color=GREEN).next_to(decomp_eq, RIGHT)
        
        display_P = DecimalMatrix(matrix_P, element_to_mobject_config={'num_decimal_places':2}).scale(0.8).next_to(p_eq, RIGHT)
        
        display_P.get_entries().set_color(GREEN)
        
        self.play(Write(p_eq), Create(display_P))
        self.wait()
        
        d_eq = MathTex(r", \space \mathbf{D} = ", color=PURPLE).next_to(display_P, RIGHT)
        
        display_D = DecimalMatrix(matrix_D, element_to_mobject_config={'num_decimal_places':2}).scale(0.8).next_to(d_eq, RIGHT)
        
        display_D.get_entries().set_color(PURPLE)
        
        self.play(Write(d_eq), Create(display_D))
        self.wait()
        
        sqrt_eq = MathTex(r"\sqrt{\mathbf{D} } = ", color=PURPLE).to_edge(LEFT).shift(DOWN*2.2)
        
        display_sqrt = DecimalMatrix(np.sqrt(matrix_D), element_to_mobject_config={'num_decimal_places':2}).scale(0.8).next_to(sqrt_eq, RIGHT)
        display_sqrt.get_entries().set_color(PURPLE)
        
        self.play(Write(sqrt_eq), Create(display_sqrt))
        self.wait()
        
        giving_eq = MathTex(r"\space , \space \text{ giving }", r"\mathbf{U}", r"=", r"\mathbf{P}", r"\sqrt{\mathbf{D} }", r"\mathbf{P}^T", r"=").scale(0.8).next_to(display_sqrt, RIGHT)
        giving_eq[1].set_color(BLUE)
        giving_eq[3].set_color(GREEN)
        giving_eq[4].set_color(PURPLE)
        giving_eq[5].set_color(GREEN)
        
        display_U = DecimalMatrix(np.matmul(matrix_P, np.matmul(np.sqrt(matrix_D), np.transpose(matrix_P))), element_to_mobject_config={'num_decimal_places':2}).scale(0.8).next_to(giving_eq, RIGHT)
        display_U.get_entries().set_color(BLUE)
        
        
        self.play(Write(giving_eq))
        self.play(Create(display_U))
        self.wait()
        
        
class SymmetricPositiveDefiniteExampleVisual(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(z_min=-4.5, z_max=4.5)
        self.set_camera_orientation(phi=60 * DEGREES, theta=-40 * DEGREES)
        self.add(axes)
        
        array_1 = [1, 0, 0]
        array_2 = [0, 3, 0]
        array_3 = [-1, -1, 1]
        
        matrix = np.transpose([np.transpose(array_1), np.transpose(array_2), np.transpose(array_3)])
        
        numerical_gramian = np.matmul(np.transpose(matrix), matrix)
        evals, matrix_P = np.linalg.eigh(numerical_gramian)
        
        gramian = DecimalMatrix(numerical_gramian, element_to_mobject_config={'num_decimal_places':2}).set_color(PURPLE).to_edge(UP).to_edge(LEFT)
        
        matrix_D = np.diag(evals)
        
        matrix_U = np.transpose(np.matmul(matrix_P, np.matmul(np.sqrt(matrix_D), np.transpose(matrix_P))))
        
        vector_1 = Arrow3D([0, 0, 0], matrix_U[0], color=BLUE)
        vector_2 = Arrow3D([0, 0, 0], matrix_U[1], color=BLUE)
        vector_3 = Arrow3D([0, 0, 0], matrix_U[2], color=BLUE)
        
        
        gramian.add_background_rectangle()
        self.add_fixed_in_frame_mobjects(gramian)
        
        self.play(Create(gramian))
        
        self.play(Create(vector_1), Create(vector_2), Create(vector_3))
        self.wait()
        
        self.begin_ambient_camera_rotation(rate=0.1)
        
        vectors = VGroup(vector_1, vector_2, vector_3)
        
        axis_1 = normalize([1, 1, 0])
        axis_2 = normalize([0, -2, 1])
        axis_3 = normalize([0, 0, 1])
        
        self.play(Rotate(vectors, axis=axis_1, about_point=[0,0,0]))
        self.wait()
        self.play(Rotate(vectors, axis=axis_2, about_point=[0,0,0]))
        self.wait()
        self.play(Rotate(vectors, axis=axis_3, about_point=[0,0,0]))
        self.wait()

